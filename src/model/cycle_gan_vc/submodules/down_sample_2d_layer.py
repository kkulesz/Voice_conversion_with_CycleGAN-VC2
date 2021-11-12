import torch
import torch.nn as nn
from src.model.cycle_gan_vc.submodules.glu import GLU


class DownSample2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample2DLayer, self).__init__()
        self.sequential = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            torch.nn.InstanceNorm2d(
                num_features=out_channels,
                affine=True
            ),
            GLU()
        )

    def forward(self, x: torch.Tensor):
        return self.sequential(x)
