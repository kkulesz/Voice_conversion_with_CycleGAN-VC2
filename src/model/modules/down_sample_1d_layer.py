from typing import Tuple

import torch
import torch.nn as nn
from src.model.modules.glu import GLU


class DownSample1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample1DLayer, self).__init__()
        self.sequential = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.InstanceNorm1d(
                num_features=out_channels,
                affine=True
            ),
            GLU()
        )

    def forward(self, x: torch.Tensor):
        return self.sequential(x)
