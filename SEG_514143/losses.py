# STUDENT's UČO: 514143

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataset import SegDataset



class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        num_classes = logits.shape[1]

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            targets = targets.clone()
            targets[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        probs = torch.softmax(logits, dim=1)

        targets_one_hot = F.one_hot( targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        if self.ignore_index is not None:
            keep_classes = torch.arange(num_classes, device=logits.device) != self.ignore_index
            dice_score = dice_score[keep_classes]

        return 1.0 - dice_score.mean()


class HybridSegmentationLoss(nn.Module):
    def __init__(self, class_weights: Tensor | None = None, ignore_index: int | None = None) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.ce(logits, targets) + self.dice(logits, targets)


def compute_class_weights(train_dataset: SegDataset, num_classes: int) -> Tensor:
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for sample in train_dataset.samples:
        mask = sample["mask"]
        class_counts += np.bincount(mask.reshape(-1), minlength=num_classes)

    class_freq = class_counts / np.maximum(class_counts.sum(), 1.0)
    weights = 1.0 / np.log(1.02 + class_freq)
    weights = weights / np.maximum(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)