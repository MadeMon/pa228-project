# STUDENT's UCO: 514143

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from metrics import (
    MetricResult,
    compute_miou_from_cm,
    update_confusion_matrix,
)
from losses import HybridSegmentationLoss, compute_class_weights
from rare_crops import MixedCropTransform
from dataset import SegDataset
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph


from config import CONFIG
from utils import create_dataframe, device, save_checkpoint

import albumentations as A

import mlflow

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
from dotenv import load_dotenv

load_dotenv()

if (
    os.getenv("DATABRICKS_MLFLOW_USERNAME")
    and os.getenv("DATABRICKS_HOST")
    and os.getenv("DATABRICKS_TOKEN")
):
    mlflow.login(backend="databricks", interactive=False)
    mlflow.set_experiment(
        f"/Users/{os.getenv('DATABRICKS_MLFLOW_USERNAME')}/segmentation_experiment"
    )

# fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def create_data_loaders(
    train_dataset: SegDataset,
    val_dataset: SegDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "drop_last": True,
    }
    val_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
    }

    if pin_memory:
        train_loader_kwargs["pin_memory"] = True
        val_loader_kwargs["pin_memory"] = True

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataloader = DataLoader(train_dataset, **train_loader_kwargs)
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)
    return train_dataloader, val_dataloader


def split_dataframe(df, train_ratio: float = 0.7, val_ratio: float = 0.2):
    total_samples = len(df)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    permutation = torch.randperm(total_samples)
    train_idx = permutation[:train_size].tolist()
    val_idx = permutation[train_size : train_size + val_size].tolist()
    test_idx = permutation[train_size + val_size :].tolist()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def autocast_context(use_amp) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# sample function for training
def fit(
    net: nn.Module,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    metrics: list[Tuple[str, Callable[[Tensor], MetricResult]]],
    main_metric_name: str = "mIoU",
    maximize_main_metric: bool = True,
    checkpoint_dir: Optional[Path] = None,
    use_amp: bool = False,
    use_grad_scaler: bool = False,
    use_non_blocking_transfer: bool = False,
    start_epoch: int = 0,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []

    best_main_metric = -float("inf") if maximize_main_metric else float("inf")

    net = net.to(device)
    loss = loss.to(device)
    scaler: torch.amp.GradScaler | None = None
    if use_grad_scaler:
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    def run_epoch(
        dataloader: DataLoader, is_train: bool
    ) -> tuple[float, torch.Tensor | None]:
        running_loss = 0.0
        batches = 0
        accumulated_cm = None

        with torch.set_grad_enabled(is_train):
            for images, targets, _ in tqdm(
                dataloader,
                desc="{} Batches".format("Training" if is_train else "Validation"),
            ):
                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                images = images.to(device, non_blocking=use_non_blocking_transfer)
                targets = targets.to(device, non_blocking=use_non_blocking_transfer)
                # Keep masks compact in host/pinned memory (uint8) and cast to long on-device.
                targets_long = targets.long()

                with autocast_context(use_amp=use_amp):
                    preds_out = net(images)

                with autocast_context(use_amp=use_amp):
                    if isinstance(preds_out, dict):
                        total_loss = loss(preds_out["out"], targets_long)
                        if "aux" in preds_out and preds_out["aux"] is not None:
                            total_loss = total_loss + 0.4 * loss(
                                preds_out["aux"], targets_long
                            )
                    else:
                        total_loss = loss(preds_out, targets_long)

                if is_train:
                    if scaler is not None:
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item()
                batches += 1

                # For validation, accumulate confusion matrix
                if not is_train:
                    preds_for_cm = (
                        preds_out["out"] if isinstance(preds_out, dict) else preds_out
                    )
                    num_classes = preds_for_cm.shape[1]
                    # Use the on-device long targets for confusion matrix accumulation.
                    accumulated_cm = update_confusion_matrix(
                        accumulated_cm,
                        preds_for_cm,
                        targets_long,
                        num_classes,
                        ignore_index=CONFIG["class_ignore_index"],
                    )

        avg_loss = running_loss / max(batches, 1)
        return avg_loss, accumulated_cm

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        net.train()

        avg_train_loss, _ = run_epoch(train_dataloader, is_train=True)
        train_losses.append(avg_train_loss)

        net.eval()
        avg_val_loss, val_confusion_matrix = run_epoch(val_dataloader, is_train=False)
        val_losses.append(avg_val_loss)

        if val_confusion_matrix is None:
            raise RuntimeError("Validation confusion matrix was not computed.")

        # print training info
        print(
            "Epoch {}, train loss: {:.5f}, val loss: {:.5f}".format(
                epoch + 1, avg_train_loss, avg_val_loss
            )
        )

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        val_metrics_results: dict[str, MetricResult] = {}

        for metric_name, metric in metrics:
            metric_result = metric(val_confusion_matrix)
            val_metrics_results[metric_name] = metric_result
            print(f"\t{metric_name}: {metric_result.main:.4f}")
            mlflow.log_metric(metric_name, metric_result.main, step=epoch)
            if metric_result.per_class is not None:
                for cls, value in metric_result.per_class.items():
                    mlflow.log_metric(f"{metric_name}_class_{cls}", value, step=epoch)
                    print(f"\t\tClass {cls}: {value:.4f}")

        # Save checkpoint every epoch
        if checkpoint_dir is not None and (
            (
                maximize_main_metric
                and val_metrics_results[main_metric_name].main > best_main_metric
            )
            or (
                not maximize_main_metric
                and val_metrics_results[main_metric_name].main < best_main_metric
            )
        ):
            unwrapped_net = getattr(net, "_orig_mod", net)
            save_checkpoint(
                unwrapped_net,
                optimizer,
                epoch,
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    **val_metrics_results,
                },
                checkpoint_dir,
                crop_size=CONFIG["transforms_random_crop_size"],
            )

        if scheduler is not None:
            scheduler.step()

    print("Training finished!")
    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    # Check for available GPU
    print("Computing with {}!".format(device))

    torch.backends.cudnn.benchmark = bool(CONFIG["runtime_cudnn_benchmark"])

    df = create_dataframe(dataset_path)

    train_df, val_df, _ = split_dataframe(df)

    # transforms
    train_transforms_list = []

    if CONFIG["transforms_random_crop_size"]:
        train_transforms_list.append(
            MixedCropTransform(
                width=CONFIG["transforms_random_crop_size"],
                height=CONFIG["transforms_random_crop_size"],
                rare_classes=CONFIG["transforms_rare_classes"],
                rare_prob=CONFIG["transforms_rare_crop_prob"],
                debug_dir=CONFIG["debug_dir_rare_crops"],
            )
        )
    else:
        train_transforms_list.append(
            A.RandomCrop(
                width=CONFIG["transforms_random_crop_size"],
                height=CONFIG["transforms_random_crop_size"],
            )
        )

    train_transforms_list.extend(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    train_transforms_list.append(
        A.Normalize(
            mean=CONFIG["transforms_normalize_mean"],
            std=CONFIG["transforms_normalize_std"],
        ),
    )

    train_transforms = A.Compose(train_transforms_list)
    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=CONFIG["transforms_normalize_mean"],
                std=CONFIG["transforms_normalize_std"],
            )
        ]
    )

    # dataset and dataloader
    train_dataset = SegDataset(
        train_df, transforms=train_transforms, preload_samples=CONFIG["preload_samples"]
    )
    val_dataset = SegDataset(
        val_df, transforms=val_transforms, preload_samples=CONFIG["preload_samples"]
    )

    train_dataloader, val_dataloader = create_data_loaders(
        train_dataset,
        val_dataset,
        CONFIG["batch_size"],
        CONFIG["runtime_num_workers"],
        CONFIG["runtime_pin_memory"],
        CONFIG["runtime_persistent_workers"],
        CONFIG["runtime_prefetch_factor"],
    )

    net = CONFIG["network"]

    # Draw the model architecture
    input_sample = torch.zeros((1, 3, 512, 1024))
    draw_network_architecture(net, input_sample)

    # optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["optimizer_weight_decay"],
    )
    num_training_steps = CONFIG["epochs"]
    warmup_steps = max(1, int(CONFIG["warmup_ratio"] * num_training_steps))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=num_training_steps - warmup_steps,
        power=0.9,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[warmup_steps],
    )

    # loss function
    class_weights = compute_class_weights(
        train_dataset, num_classes=len(train_dataset.color_to_class)
    )
    loss = HybridSegmentationLoss(
        class_weights=class_weights, ignore_index=CONFIG["class_ignore_index"]
    )

    # metrics
    metrics: list[Any] = [
        (
            "mIoU",
            lambda cm: compute_miou_from_cm(
                cm, ignore_index=CONFIG["class_ignore_index"]
            ),
        )
    ]

    # Try to resume from latest checkpoint
    start_epoch = 0
    try:
        from utils import load_checkpoint

        _, start_epoch = load_checkpoint(
            CONFIG["model_checkpoint_path"],
            net,
            optimizer,
            crop_size=CONFIG["transforms_random_crop_size"],
        )
        print(f"Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

    # compile the model
    train_net: nn.Module = net
    if CONFIG["runtime_compile_model"]:
        if hasattr(torch, "compile"):
            try:
                train_net = cast(nn.Module, torch.compile(net))
                print("Enabled torch.compile for CUDA training.")
            except Exception as exc:
                print(f"torch.compile failed, using eager mode: {exc}")
        else:
            print(
                "torch.compile is not available in this PyTorch build; using eager mode."
            )

    # log config to mlflow
    print("CONFIG:")
    for key, value in CONFIG.items():
        print(f"\t{key}: {value}")
    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        # train the network
        train_losses, val_losses = fit(
            train_net,
            CONFIG["epochs"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            metrics=metrics,
            maximize_main_metric=True,
            checkpoint_dir=CONFIG["model_checkpoint_path"],
            use_amp=CONFIG["runtime_amp"],
            use_grad_scaler=CONFIG["runtime_grad_scaler"],
            use_non_blocking_transfer=CONFIG["runtime_non_blocking_transfer"],
            start_epoch=start_epoch,
        )

    # save the trained model and plot the losses
    torch.save(net.state_dict(), "model.pt")
    plot_learning_curves(train_losses, val_losses)


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
