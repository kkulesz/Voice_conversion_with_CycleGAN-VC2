import torch
import torch.nn as nn

import numpy as np

from src.modules.submodules.glu import GLU
from src.modules.submodules.down_sample_1d_layer import DownSample1DLayer
from src.modules.submodules.up_sample import UpSampleLayer
from src.modules.submodules.residual_block import ResidualBlock


# TODO: find out what padding is for and decide its value
#  for now it's copied from pytorch followup work but i dont know where (s)he got it from
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=128, kernel_size=15, stride=1, padding=7),
            GLU()
        )

        self.down_sample_1 = DownSample1DLayer(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.down_sample_2 = DownSample1DLayer(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)

        # TODO: probably need to create 6 residual blocks instead of iterating 6 times through the same
        self.residual_block = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.number_of_residual_block_iteration = 6

        self.up_sample_1 = UpSampleLayer(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.up_sample_2 = UpSampleLayer(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.last_conv = nn.Conv1d(in_channels=256, out_channels=24, kernel_size=15, stride=1, padding=7)

    def forward(self, x):
        after_initial_conv = self.initial_conv(x)
        after_first_down_sampling = self.down_sample_1(after_initial_conv)
        after_down_sampling = self.down_sample_2(after_first_down_sampling)

        for_residual_blocks = after_down_sampling
        for i in range(self.number_of_residual_block_iteration):
            for_residual_blocks = self.residual_block(for_residual_blocks)
        after_residual_block = for_residual_blocks

        after_first_up_sampling = self.up_sample_1(after_residual_block)
        after_up_sampling = self.up_sample_2(after_first_up_sampling)

        result = self.last_conv(after_up_sampling)
        return result


if __name__ == '__main__':
    # Dimensionality Testing
    generator = Generator()

    x = torch.randn(158, 24, 128)
    output = generator(x)
    print("Generator-output shape:", output.shape)