import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.up_sample_generator import UpSampleGenerator
from src.model.cycle_gan_vc2.submodules.down_sample_generator import DownSampleGenerator
from src.model.cycle_gan_vc2.submodules.residual_block import ResidualLayer


class GeneratorCycleGan2(nn.Module):
    def __init__(self):
        super(GeneratorCycleGan2, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))
        self.conv_gates = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=1, padding=(2, 7))

        self.down_sample_1 = DownSampleGenerator(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.down_sample_2 = DownSampleGenerator(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)

        # 2D -> 1D
        self.conv_2d_to_1d = nn.Sequential(
            nn.Conv1d(in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        self.rd1 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.rd2 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.rd3 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.rd4 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.rd5 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.rd6 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        # 1D -> 2D
        self.conv_1d_to_2d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=2304, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm1d(num_features=2304, affine=True)
        )

        self.up_sample_1 = UpSampleGenerator(in_channels=256, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.up_sample_2 = UpSampleGenerator(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.output_conv = \
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))

    def forward(self, x):
        unsqueezed = x.unsqueeze(1)
        after_conv = self.conv(unsqueezed) * torch.sigmoid(self.conv_gates(unsqueezed))

        after_down_sample_1 = self.down_sample_1(after_conv)
        after_down_sample_2 = self.down_sample_2(after_down_sample_1)

        # 2D -> 1D
        reshaped_to_1d = after_down_sample_2.view(after_down_sample_2.size(0), 2304, 1, -1)
        reshaped_to_1d = reshaped_to_1d.squeeze(2)
        after_conv2dto1d = self.conv_2d_to_1d(reshaped_to_1d)

        rd1 = self.rd1(after_conv2dto1d)
        rd2 = self.rd2(rd1)
        rd3 = self.rd3(rd2)
        rd4 = self.rd4(rd3)
        rd5 = self.rd5(rd4)
        rd6 = self.rd6(rd5)

        # 1D -> 2D
        after_conv1dto2d = self.conv_1d_to_2d(rd6)
        reshaped_to_2d = after_conv1dto2d.unsqueeze(2)
        reshaped_to_2d = reshaped_to_2d.view(reshaped_to_2d.size(0), 256, 9, -1)

        after_up_sample_1 = self.up_sample_1(reshaped_to_2d)
        after_up_sample_2 = self.up_sample_2(after_up_sample_1)

        after_output_conv = self.output_conv(after_up_sample_2)
        return after_output_conv.squeeze(1)
