# STUDENT's UCO: 514143

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import datetime
from pathlib import Path
import random
from typing import Any, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time
from losses import HybridSegmentationLoss, compute_class_weights
from rare_crops import MixedCropTransform
from dataset import SegDataset
from network import ModelLRASPP
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph


from utils import create_dataframe, device, save_checkpoint

import albumentations as A

import mlflow

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from dotenv import load_dotenv

load_dotenv()

mlflow.login(backend="databricks", interactive=False)
mlflow.set_experiment(f"/Users/{os.getenv('DATABRICKS_MLFLOW_USERNAME')}/segmentation_experiment")    

# fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


TRAINING_SCHEDULER: Any | None = None

CONFIG: dict[str, Any] = {
    "batch_size": 32,
    "epochs": 500,
    "learning_rate": 1e-3,
    "optimizer_weight_decay": 1e-4,
    "transforms_random_crop_size": 256,  # 512, 256, None
    "transforms_rare_crop_prob": 0.5,
    "rare_classes": [
        3,  # object
        6,  # human
    ],
    "debug_dir_rare_crops": None,  # "debug_rare_crops", None
    "class_ignore_index": 0,  # 0 is the "void" class
    "test_mode": False,  # use subset of 10 samples to test the training pipeline - try to overfit the model on this tiny dataset, if it doesn't work, there is likely a bug in the training pipeline
    "model_checkpoint_path": "models",
    "cuda_use_amp": True,
    "cuda_use_grad_scaler": True,
    "cuda_pin_memory": True,
    "cuda_non_blocking_transfer": True,
    "cuda_num_workers": 4,
    "other_num_workers": 0,
    "cuda_prefetch_factor": 2,
    "cuda_persistent_workers": True,
    "cuda_cudnn_benchmark": True,
    "cuda_compile_model": False,
    "log_runtime_metadata": True,
    "preload_samples": device.type == "mps",  # Preload samples into memory for faster access on MPS, where disk I/O is slow. 
}


def create_data_loaders(
    train_dataset: SegDataset,
    val_dataset: SegDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    is_cuda: bool,
) -> Tuple[DataLoader, DataLoader, dict[str, Any], dict[str, Any]]:
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
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
        persistent_workers = bool(is_cuda and CONFIG["cuda_persistent_workers"])
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers
        if is_cuda:
            train_loader_kwargs["prefetch_factor"] = CONFIG["cuda_prefetch_factor"]
            val_loader_kwargs["prefetch_factor"] = CONFIG["cuda_prefetch_factor"]

    train_dataloader = DataLoader(train_dataset, **train_loader_kwargs)
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)
    return train_dataloader, val_dataloader, train_loader_kwargs, val_loader_kwargs


@dataclass
class MetricResult:
    main: float
    per_class: dict[int, float] | None = None


def aggregate_metric_results(results: list["MetricResult"]) -> "MetricResult":
    main = float(np.mean([r.main for r in results]))
    if results and results[0].per_class is not None:
        all_classes = results[0].per_class.keys()
        per_class = {
            cls: float(np.mean([r.per_class[cls] for r in results if cls in r.per_class]))
            for cls in all_classes
        }
    else:
        per_class = None
    return MetricResult(main=main, per_class=per_class)

def split_dataframe(df, train_ratio: float = 0.7, val_ratio: float = 0.2):
    total_samples = len(df)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    permutation = torch.randperm(total_samples)
    train_idx = permutation[:train_size].tolist()
    val_idx = permutation[train_size:train_size + val_size].tolist()
    test_idx = permutation[train_size + val_size:].tolist()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df

def compute_mlou_from_cm(
    confusion_matrix: Tensor | np.ndarray, ignore_index: int | None = None
) -> MetricResult:
    """Compute mean IoU from accumulated confusion matrix.
    
    Args:
        confusion_matrix: Accumulated confusion matrix of shape (num_classes, num_classes)
        ignore_index: Class index to ignore when computing mean
    
    Returns:
        MetricResult with mean IoU and per-class IoU dict
    """
    if isinstance(confusion_matrix, torch.Tensor):
        cm = confusion_matrix.detach().cpu().numpy()
    else:
        cm = confusion_matrix

    num_classes = cm.shape[0]
    ious: dict[int, float] = {}
    
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        
        # TP: diagonal element, FP: sum of column minus TP, FN: sum of row minus TP
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        
        denominator = tp + fp + fn
        if denominator > 0:
            ious[cls] = float(tp / denominator)
        else:
            ious[cls] = 0.0
    
    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    return MetricResult(main=mean_iou, per_class=ious)


def plot_confusion_matrix(
    confusion_matrix: Tensor | np.ndarray, num_classes: int, class_names: list[str] | None = None
) -> plt.Figure:
    """Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        num_classes: Number of classes
        class_names: Optional list of class names for labels
    
    Returns:
        Matplotlib figure object
    """
    if isinstance(confusion_matrix, torch.Tensor):
        cm = confusion_matrix.detach().cpu().numpy()
    else:
        cm = confusion_matrix

    # Normalize CM by row (per true class) for better visualization
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm_normalized / row_sums
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, cmap="Blues", aspect="auto")
    
    # Set ticks and labels
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix (Normalized by True Class)")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations for raw counts
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(
                j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center", va="center", color="black", fontsize=8
            )
    
    fig.tight_layout()
    return fig

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


@torch.no_grad()
def update_confusion_matrix(
    cm: torch.Tensor | None,
    preds_logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    # argmax is fast on both CUDA and MPS — keep it on-device.
    preds = preds_logits.argmax(dim=1).reshape(-1)
    targets_flat = targets.reshape(-1)

    if preds_logits.device.type != "cuda":
        # torch.bincount is poorly supported on MPS; move the small class-index
        # tensors (not the full logit tensor) to CPU where bincount is native.
        preds = preds.cpu()
        targets_flat = targets_flat.cpu()

    preds = preds.to(torch.int64)
    targets_flat = targets_flat.to(torch.int64)

    if ignore_index is not None:
        valid = targets_flat != ignore_index
        linear_idx = targets_flat * num_classes + preds
        # Route ignored pixels to an extra overflow bin so bincount stays branchless.
        ignore_bin = num_classes * num_classes
        linear_idx[~valid] = ignore_bin
        counts = torch.bincount(linear_idx, minlength=ignore_bin + 1)
        batch_cm = counts[:ignore_bin].reshape(num_classes, num_classes)
    else:
        linear_idx = targets_flat * num_classes + preds
        batch_cm = torch.bincount(
            linear_idx,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if cm is None:
        return batch_cm

    cm.add_(batch_cm)
    return cm


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
    metrics: list[Callable[[Tensor, Tensor], dict[str, float]]] | None = None,
    checkpoint_path: Optional[Path] = None,
    use_amp: bool = False,
    use_grad_scaler: bool = False,
    use_non_blocking_transfer: bool = False,
) -> tuple[list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")

    net = net.to(device)
    loss = loss.to(device)
    use_amp = bool(use_amp and device.type == "cuda")
    use_grad_scaler = bool(use_grad_scaler and use_amp and device.type == "cuda")
    scaler: torch.cuda.amp.GradScaler | None = None
    if use_grad_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    def autocast_context() -> Any:
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def run_epoch(
        dataloader: DataLoader, is_train: bool
    ) -> tuple[
        float, dict[str, list[MetricResult]], torch.Tensor | None, dict[str, float]
    ]:
        running_loss = 0.0
        batches = 0
        metrics_results: dict[str, list[MetricResult]] = {}
        for metric_fn in metrics or []:
            metrics_results[metric_fn.__name__] = []

        # Initialize confusion matrix for validation (will be properly sized after first batch)
        accumulated_cm = None

        # Timing accumulators (only used for validation)
        total_move_time = 0.0
        total_forward_time = 0.0
        total_loss_time = 0.0
        total_cm_time = 0.0
        total_other_time = 0.0

        with torch.set_grad_enabled(is_train):
            for images, targets in tqdm(dataloader, desc="{} Batches".format("Training" if is_train else "Validation")):
                batch_start = time.perf_counter()

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # Measure device transfer time
                t0 = time.perf_counter()
                images = images.to(device, non_blocking=use_non_blocking_transfer)
                targets = targets.to(device, non_blocking=use_non_blocking_transfer)
                t1 = time.perf_counter()

                # Forward pass
                t2 = time.perf_counter()
                with autocast_context():
                    preds_out = net(images)
                t3 = time.perf_counter()

                # Loss computation (and auxiliary loss if present)
                t4 = time.perf_counter()
                with autocast_context():
                    if isinstance(preds_out, dict):
                        total_loss = loss(preds_out["out"], targets)
                        if "aux" in preds_out and preds_out["aux"] is not None:
                            total_loss = total_loss + 0.4 * loss(
                                preds_out["aux"], targets
                            )
                    else:
                        total_loss = loss(preds_out, targets)
                t5 = time.perf_counter()

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
                    preds_for_cm = preds_out["out"] if isinstance(preds_out, dict) else preds_out
                    num_classes = preds_for_cm.shape[1]

                    t6 = time.perf_counter()
                    accumulated_cm = update_confusion_matrix(accumulated_cm, preds_for_cm, targets, num_classes, ignore_index=CONFIG["class_ignore_index"])
                    t7 = time.perf_counter()

                    # accumulate timings
                    total_move_time += (t1 - t0)
                    total_forward_time += (t3 - t2)
                    total_loss_time += (t5 - t4)
                    total_cm_time += (t7 - t6)
                    total_other_time += (time.perf_counter() - batch_start) - ((t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6))

        avg_loss = running_loss / max(batches, 1)

        # Prepare average timings per batch for validation
        timings: dict[str, float] = {}
        if not is_train and batches > 0:
            timings = {
                "avg_move_time": total_move_time / batches,
                "avg_forward_time": total_forward_time / batches,
                "avg_loss_time": total_loss_time / batches,
                "avg_cm_time": total_cm_time / batches,
                "avg_other_time": total_other_time / batches,
                "total_batches": float(batches),
            }
        return avg_loss, metrics_results, accumulated_cm, timings

    for epoch in tqdm(range(epochs), desc="Epochs"):
        net.train()

        avg_train_loss, _, _, _ = run_epoch(train_dataloader, is_train=True)
        train_losses.append(avg_train_loss)

        net.eval()
        avg_val_loss, val_metrics_results, val_confusion_matrix, val_timings = run_epoch(val_dataloader, is_train=False)
        val_losses.append(avg_val_loss)

        if checkpoint_path is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(net, optimizer, epoch, {"train_loss": avg_train_loss, "val_loss": avg_val_loss, **val_metrics_results}, checkpoint_path)

        # print training info
        print(
            "Epoch {}, train loss: {:.5f}, val loss: {:.5f}".format(
                epoch + 1, avg_train_loss, avg_val_loss
            )
        )

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Compute and log mIoU from accumulated confusion matrix
        if val_confusion_matrix is not None:
            mlou_result = compute_mlou_from_cm(val_confusion_matrix, ignore_index=CONFIG["class_ignore_index"])
            print("Validation metrics:")
            print(f"\tmIoU: {mlou_result.main:.4f}")
            mlflow.log_metric("mIoU", mlou_result.main, step=epoch)

            if mlou_result.per_class is not None:
                for cls, value in mlou_result.per_class.items():
                    mlflow.log_metric(f"mIoU_class_{cls}", value, step=epoch)
                    print(f"\t\tClass {cls}: {value:.4f}")

            # Plot and log confusion matrix
            cm_fig = plot_confusion_matrix(val_confusion_matrix, num_classes=val_confusion_matrix.shape[0])
            mlflow.log_figure(cm_fig, f"confusion_matrix_epoch_{epoch}.png")
            plt.close(cm_fig)

            # Save confusion matrix as numpy
            cm_npy_path = f"confusion_matrix_epoch_{epoch}.npy"
            if isinstance(val_confusion_matrix, torch.Tensor):
                cm_to_save = val_confusion_matrix.detach().cpu().numpy()
            else:
                cm_to_save = val_confusion_matrix
            np.save(cm_npy_path, cm_to_save)
            mlflow.log_artifact(cm_npy_path)

            # Log validation timing breakdown
            if val_timings:
                print("Validation timing (avg seconds per batch):")
                for k, v in val_timings.items():
                    if k == "total_batches":
                        continue
                    print(f"\t{k}: {v:.6f}s")
                    mlflow.log_metric(f"val_timing_{k}", float(v), step=epoch)

        # print other metrics info (if any additional metrics are added)
        if metrics is not None and val_metrics_results:
            for metric_name, batch_results in val_metrics_results.items():
                agg = aggregate_metric_results(batch_results)
                print(f"\t{metric_name}: {agg.main:.4f}")
                mlflow.log_metric(metric_name, agg.main, step=epoch)
                if agg.per_class is not None:
                    for cls, value in agg.per_class.items():
                        mlflow.log_metric(f"{metric_name}_class_{cls}", value, step=epoch)
                        print(f"\t\tClass {cls}: {value:.4f}")

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

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(CONFIG["cuda_cudnn_benchmark"])

    df = create_dataframe(dataset_path)

    train_df, val_df, _ = split_dataframe(df)

    if CONFIG["test_mode"]:
        train_df = train_df.head(10)
        val_df = train_df

    # transforms
    train_transforms_list = []

    if not CONFIG["test_mode"]:
        if CONFIG["transforms_random_crop_size"]:
            train_transforms_list.append(
                MixedCropTransform(
                    width=CONFIG["transforms_random_crop_size"],
                    height=CONFIG["transforms_random_crop_size"],
                    rare_classes=CONFIG["rare_classes"],
                    rare_prob=CONFIG["transforms_rare_crop_prob"],
                    debug_dir=CONFIG["debug_dir_rare_crops"]
                )
            )
        else:
            train_transforms_list.append(
                A.RandomCrop(width=CONFIG["transforms_random_crop_size"], height=CONFIG["transforms_random_crop_size"])
            )

        train_transforms_list.extend([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=CONFIG["class_ignore_index"],
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),

            # Alternative to ColorJitter
            # A.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.05,
            #     p=0.3,
            # ),
        ])

    train_transforms_list.append(
        A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
    )

    train_transforms = A.Compose(train_transforms_list)
    val_transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))])

    # dataset and dataloader
    train_dataset = SegDataset(train_df, transforms=train_transforms, preload_samples=CONFIG["preload_samples"])
    val_dataset = SegDataset(val_df, transforms=val_transforms, preload_samples=CONFIG["preload_samples"])

    is_cuda = device.type == "cuda"
    num_workers = CONFIG["cuda_num_workers"] if is_cuda else CONFIG["other_num_workers"]
    pin_memory = bool(is_cuda and CONFIG["cuda_pin_memory"])

    train_dataloader, val_dataloader, train_loader_kwargs, val_loader_kwargs = (
        create_data_loaders(
            train_dataset,
            val_dataset,
            CONFIG["batch_size"],
            num_workers,
            pin_memory,
            is_cuda,
        )
    )

    net = ModelLRASPP()

    # Draw the model architecture
    input_sample = torch.zeros((1, 3, 512, 1024))
    draw_network_architecture(net, input_sample)

    train_net: nn.Module = net
    compile_enabled = False
    if is_cuda and CONFIG["cuda_compile_model"]:
        if hasattr(torch, "compile"):
            try:
                train_net = cast(nn.Module, torch.compile(net))
                compile_enabled = True
                print("Enabled torch.compile for CUDA training.")
            except Exception as exc:
                print(f"torch.compile failed, using eager mode: {exc}")
        else:
            print(
                "torch.compile is not available in this PyTorch build; using eager mode."
            )

    class_weights = compute_class_weights(train_dataset, num_classes=len(train_dataset.color_to_class))

    # optimizer and learning rate
    optimizer = torch.optim.AdamW(
        train_net.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["optimizer_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # loss function
    loss = HybridSegmentationLoss(class_weights=class_weights, ignore_index=CONFIG["class_ignore_index"])

    # metrics
    # Note: mIoU is now computed from accumulated confusion matrix during validation
    metrics: list[Any] = []

    # checkpoint path
    checkpoint_path = Path(f"{CONFIG['model_checkpoint_path']}/{datetime.datetime.now().strftime('%m%d_%H%M%S')}") if CONFIG["model_checkpoint_path"] else None

    use_amp = bool(is_cuda and CONFIG["cuda_use_amp"])
    use_grad_scaler = bool(use_amp and CONFIG["cuda_use_grad_scaler"])
    use_non_blocking_transfer = bool(
        is_cuda and pin_memory and CONFIG["cuda_non_blocking_transfer"]
    )

    runtime_metadata: dict[str, Any] = {
        "runtime_device": device.type,
        "runtime_amp": use_amp,
        "runtime_grad_scaler": use_grad_scaler,
        "runtime_compile": compile_enabled,
        "runtime_num_workers": num_workers,
        "runtime_pin_memory": pin_memory,
        "runtime_persistent_workers": train_loader_kwargs.get(
            "persistent_workers", False
        ),
        "runtime_prefetch_factor": train_loader_kwargs.get("prefetch_factor", "none"),
        "runtime_cudnn_benchmark": torch.backends.cudnn.benchmark if is_cuda else "n/a",
    }

    print("Runtime metadata:")
    for key, value in runtime_metadata.items():
        print(f"\t{key}: {value}")

    with mlflow.start_run():
        if CONFIG["log_runtime_metadata"]:
            for key, value in runtime_metadata.items():
                mlflow.log_param(key, value)

        # train the network
        train_losses, val_losses = fit(
            train_net,
            CONFIG["epochs"],
            train_dataloader,
            val_dataloader,
            loss,
            optimizer,
            scheduler,
            device,
            metrics,
            checkpoint_path=checkpoint_path,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
            use_non_blocking_transfer=use_non_blocking_transfer,
        )

    # save the trained model and plot the losses, feel free to create your own functions
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
