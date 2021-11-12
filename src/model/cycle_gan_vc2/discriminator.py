import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.glu import GLU
from src.model.cycle_gan_vc2.submodules.down_sample_discriminator import DownSampleDiscriminator


# PatchGAN
class DiscriminatorCycleGan2(nn.Module):
    def __init__(self):
        super(DiscriminatorCycleGan2, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            GLU()
        )

        self.down_sample1 = \
            DownSampleDiscriminator(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample2 = \
            DownSampleDiscriminator(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample3 = \
            DownSampleDiscriminator(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample4 = \
            DownSampleDiscriminator(in_channels=1024, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))

        self.output_conv = \
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        after_initial_conv = self.initial_conv(x)

        after_down_sample_1 = self.down_sample1(after_initial_conv)
        after_down_sample_2 = self.down_sample2(after_down_sample_1)
        after_down_sample_3 = self.down_sample3(after_down_sample_2)

        after_output_conv = self.output_conv(after_down_sample_3)

        return torch.sigmoid(after_output_conv)
