# pixel cnn: 3x3 dilated convolutions without padding. a patch the size of the view gives one
# output (training), a padded whole image gives a class for every pixel (mapping).
# view (receptive field) = 1 + 2 * sum(dilations) pixels.

from __future__ import annotations

import torch
from torch import nn

DILATIONS = (1, 1, 2, 4, 2, 1)


def receptive_field(dilations=DILATIONS) -> int:
    return 1 + 2 * sum(dilations)


class PixelFCN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 48, dropout: float = 0.1,
                 dilations=DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.rf = receptive_field(dilations)
        layers, ch = [], in_ch
        for d in dilations:
            layers += [nn.Conv2d(ch, width, 3, padding=0, dilation=d, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            ch = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(width, n_classes, 1))
        # per-channel input normalisation, stored with the weights so mapping uses the same
        self.register_buffer("mean", torch.zeros(in_ch))
        self.register_buffer("std", torch.ones(in_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] -> class scores [N, K, H - (rf - 1), W - (rf - 1)]."""
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return self.head(self.body(x))
