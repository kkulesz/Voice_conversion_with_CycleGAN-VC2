import torch.nn as nn
import torch


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()
        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=out_channels, affine=True)
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=out_channels, affine=True)
        )

        self.conv1d_out_layer = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(num_features=in_channels, affine=True)
        )

    def forward(self, x):
        h1_glu = self.conv1d_layer(x) * torch.sigmoid(self.conv_layer_gates(x))
        h2_norm = self.conv1d_out_layer(h1_glu)
        return x + h2_norm
