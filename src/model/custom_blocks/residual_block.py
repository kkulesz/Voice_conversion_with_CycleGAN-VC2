import torch.nn as nn
from src.model.custom_blocks.glu import GLU


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding),
            nn.InstanceNorm1d(
                num_features=out_channels,
                affine=True),
            GLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding),
            nn.InstanceNorm1d(
                num_features=in_channels,
                affine=True)
        )

    def forward(self, x):
        return x + self.sequential(x)
