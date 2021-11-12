import torch.nn as nn
import torch


class DownSampleGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleGenerator, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True)
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True)
        )

    def forward(self, x):
        return self.conv_layer(x) * torch.sigmoid(self.conv_layer_gates(x))
