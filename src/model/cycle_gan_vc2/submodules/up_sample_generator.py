import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.glu import GLU


class UpSampleGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PixelShuffle(upscale_factor=2),
            nn.InstanceNorm2d(num_features=out_channels // 4, affine=True),
            GLU()
        )

    def forward(self, x):
        return self.up_sample(x)
