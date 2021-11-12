import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.up_sample_generator import UpSampleGenerator
from src.model.cycle_gan_vc2.submodules.down_sample_generator import DownSampleGenerator
from src.model.cycle_gan_vc2.submodules.residual_block import ResidualLayer
from src.model.cycle_gan_vc2.submodules.glu import GLU


class GeneratorCycleGan2(nn.Module):
    def __init__(self):
        super(GeneratorCycleGan2, self).__init__()

        # 2D Conv Layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))
        self.conv_gates = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=1, padding=(2, 7))

        # 2D Downsample Layer
        self.downSample1 = DownSampleGenerator(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.downSample2 = DownSampleGenerator(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Sequential(
            nn.Conv1d(in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        self.residualLayer1 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=2304, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=2304, affine=True)
        )

        self.upSample1 = UpSampleGenerator(in_channels=256, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.upSample2 = UpSampleGenerator(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.output_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))

    def forward(self, x):
        x = x.unsqueeze(1)
        conv1 = self.conv(x) * torch.sigmoid(self.conv_gates(x))

        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # 2D -> 1D
        # reshape
        reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)

        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)

        # reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 9, -1)

        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        output = self.output_conv(upsample_layer_2)
        output = output.squeeze(1)
        return output
