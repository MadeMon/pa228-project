from dataclasses import dataclass

import torch

from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MetricResult:
    main: float
    per_class: dict[int, float] | None = None


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


def compute_miou_from_cm(
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


def aggregate_metric_results(results: list["MetricResult"]) -> "MetricResult":
    main = float(np.mean([r.main for r in results]))
    if results and results[0].per_class is not None:
        all_classes = results[0].per_class.keys()
        per_class = {
            cls: float(
                np.mean([r.per_class[cls] for r in results if cls in r.per_class])
            )
            for cls in all_classes
        }
    else:
        per_class = None
    return MetricResult(main=main, per_class=per_class)


def plot_confusion_matrix(
    confusion_matrix: Tensor | np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
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
                j,
                i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig
