# STUDENT's UCO: 514143

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

from torch import Tensor, nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

from label_dict import label_dict


class ModelLRASPP(nn.Module):
    def __init__(self, num_classes: int = len(label_dict)) -> None:
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            # weights_backbone = None,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.model(x)
