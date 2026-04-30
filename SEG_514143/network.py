# STUDENT's UCO: 514143

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
from torch.nn import functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1),
                nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
                nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
                nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        out = [b(x) for b in self.branches]
        out.append(
            F.interpolate(
                self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False
            )
        )
        return self.project(torch.cat(out, dim=1))


class MobilnetASPP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        )
        # high-level features from MobileNetV3 large: 960 channels at 1/32
        self.encoder = backbone.features

        self.aspp = ASPP(in_channels=960, out_channels=256)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        features = self.encoder(x)
        features = self.aspp(features)
        features = self.classifier(features)
        return F.interpolate(
            features, size=(h, w), mode="bilinear", align_corners=False
        )
