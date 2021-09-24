from typing import Tuple

import torch
import torch.nn as nn
from src.model.submodules.glu import GLU
from src.model.submodules.pixel_shuffle import PixelShuffle


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpSampleLayer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            PixelShuffle(upscale_factor=2),
            nn.InstanceNorm1d(
                num_features=out_channels // 2,
                affine=True
            ),
            GLU()
        )

    def forward(self, x):
        return self.sequential(x)
