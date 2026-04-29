# STUDENT's UČO: 514143

import os
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import mlflow
import numpy as np
from pandas import DataFrame
import torch

from metrics import plot_confusion_matrix

IMG_DIR = "img"
MASK_DIR = "mask"

IMG_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_color.png"


def sample_id_from_name(path: Path, expected_suffix: str) -> str:
    if not path.name.endswith(expected_suffix):
        raise ValueError(f"Unexpected filename format: {path.name}")
    return path.name[: -len(expected_suffix)]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def create_dataframe(dataset_path) -> DataFrame:
    """Creates a dataframe from the given dataset path.

    Args:
        dataset_path: Path to the dataset.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    data_path = dataset_path / IMG_DIR

    city_folders = [folder.name for folder in data_path.iterdir() if folder.is_dir()]

    samples = []

    for city_folder in city_folders:
        img_folder = dataset_path / IMG_DIR / city_folder
        mask_folder = dataset_path / MASK_DIR / city_folder

        img_files = list(img_folder.glob("*.png"))

        for img_file in img_files:
            img_id = sample_id_from_name(img_file, IMG_SUFFIX)
            mask_file = mask_folder / f"{img_id}{MASK_SUFFIX}"
            samples.append(
                {
                    "img_path": img_file,
                    "mask_path": mask_file,
                }
            )
    return DataFrame(samples)


def create_dataframe_tiny_set(dataset_path, num_samples: int = 10) -> DataFrame:
    """Creates a tiny dataframe from the given dataset path.
    This function is used for testing purposes.

    Args:
        dataset_path: Path to the dataset.
        num_samples: Number of samples to include in the tiny dataset.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    data_path = dataset_path / IMG_DIR

    city_folders = [folder.name for folder in data_path.iterdir() if folder.is_dir()]

    samples = []

    for city_folder in city_folders:
        img_folder = dataset_path / IMG_DIR / city_folder
        mask_folder = dataset_path / MASK_DIR / city_folder

        img_files = list(img_folder.glob("*.png"))

        for img_file in img_files:
            img_id = sample_id_from_name(img_file, IMG_SUFFIX)
            mask_file = mask_folder / f"{img_id}{MASK_SUFFIX}"
            samples.append(
                {
                    "img_path": img_file,
                    "mask_path": mask_file,
                }
            )
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    return DataFrame(samples)


def compose_checkpoint_path(checkpoint_dir: Path, model, optimizer, crop_size) -> Path:
    """Compose a checkpoint path based on the model, optimizer, and crop size."""
    path = (
        checkpoint_dir
        / f"{model.__class__.__name__}_opt_{optimizer.__class__.__name__}_crop_{crop_size}_best"
    )
    return path.with_suffix(".pt")


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir: Path, crop_size):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = compose_checkpoint_path(checkpoint_dir, model, optimizer, crop_size)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_class": model.__class__.__name__,
        },
        path,
    )


def load_checkpoint(
    checkpoint_dir: Path | None, model, optimizer, crop_size
) -> tuple[dict[str, Any] | None, int]:
    """Load model checkpoint."""
    if checkpoint_dir is None:
        raise ValueError(
            "No checkpoint directory provided, skipping checkpoint loading."
        )

    path = compose_checkpoint_path(checkpoint_dir, model, optimizer, crop_size)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint, checkpoint["epoch"]


def plot_and_save_confusion_matrix(
    cm: torch.Tensor, class_names: list[str], epoch: int
) -> None:
    """Plot and save confusion matrix."""
    # Plot and log confusion matrix
    cm_fig = plot_confusion_matrix(
        cm, num_classes=len(class_names), class_names=class_names
    )
    mlflow.log_figure(cm_fig, f"confusion_matrix_epoch_{epoch}.png")
    plt.close(cm_fig)

    # Save confusion matrix as numpy
    cm_npy_path = f"confusion_matrix_epoch_{epoch}.npy"
    if isinstance(cm, torch.Tensor):
        cm_to_save = cm.detach().cpu().numpy()
    else:
        cm_to_save = cm
    np.save(cm_npy_path, cm_to_save)
    mlflow.log_artifact(cm_npy_path)
    os.remove(cm_npy_path)
