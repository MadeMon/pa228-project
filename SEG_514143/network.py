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
        # Split encoder at stride-8 boundary (features[6] → 40ch at 1/8)
        # features[7:] takes it from 40ch/stride-8 to 960ch/stride-32
        self.encoder_low = backbone.features[:7]   # 40ch, stride 8
        self.encoder_high = backbone.features[7:]  # 960ch, stride 32

        # Project low-level features to fewer channels before fusion
        self.low_level_project = nn.Sequential(
            nn.Conv2d(40, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.aspp = ASPP(in_channels=960, out_channels=256)

        # Decoder: refines fused high-level + low-level features (DeepLabV3+ style)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        # Encoder: split at stride-8
        low_level_raw = self.encoder_low(x)           # B×40×H/8×W/8
        high_level = self.encoder_high(low_level_raw) # B×960×H/32×W/32

        # ASPP on high-level context features
        high_level = self.aspp(high_level)            # B×256×H/32×W/32

        # Upsample ASPP output to low-level feature resolution
        lh, lw = low_level_raw.shape[2:]
        high_level = F.interpolate(
            high_level, size=(lh, lw), mode="bilinear", align_corners=False
        )                                             # B×256×H/8×W/8

        # Project and fuse
        low_level = self.low_level_project(low_level_raw)    # B×48×H/8×W/8
        fused = torch.cat([high_level, low_level], dim=1)    # B×304×H/8×W/8

        # Decode and classify
        out = self.decoder(fused)                     # B×256×H/8×W/8
        out = self.classifier(out)                    # B×C×H/8×W/8
        return F.interpolate(
            out, size=(h, w), mode="bilinear", align_corners=False
        )
