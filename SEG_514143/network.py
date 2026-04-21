# STUDENT's UCO: 514143

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch.nn.functional as F
from torch import Tensor, nn


class ModelExample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
