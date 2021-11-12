import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=out_channels, affine=True)
        )

        self.conv_gates = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=out_channels, affine=True)
        )

        self.output_conv = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=in_channels, affine=True)
        )

    def forward(self, x):
        after_initial_conv_glu = self.conv(x) * torch.sigmoid(self.conv_gates(x))
        after_output_conv = self.output_conv(after_initial_conv_glu)
        return x + after_output_conv
