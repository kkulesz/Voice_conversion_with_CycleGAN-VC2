import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.glu import GLU


class DownSampleDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            GLU()
        )

    def forward(self, x):
        return self.down_sample(x)
