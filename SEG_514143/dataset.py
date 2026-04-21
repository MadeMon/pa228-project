# STUDENT's UCO: 514143

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SampleDataset(Dataset[Tensor]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tensor:
        return torch.zeros((256, 256))
