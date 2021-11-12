import torch.nn as nn
import torch

from src.model.cycle_gan_vc2.submodules.glu import GLU
from src.model.cycle_gan_vc2.submodules.down_sample_discriminator import DownSampleDiscriminator


# PatchGAN
class DiscriminatorCycleGan2(nn.Module):
    def __init__(self):
        super(DiscriminatorCycleGan2, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            GLU()
        )

        self.down_sample1 = DownSampleDiscriminator(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample2 = DownSampleDiscriminator(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample3 = DownSampleDiscriminator(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down_sample4 = DownSampleDiscriminator(in_channels=1024, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))

        self.output_conv = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def forward(self, x):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        x = x.unsqueeze(1)
        conv_layer_1 = self.conv_layer(x)

        downsample1 = self.down_sample1(conv_layer_1)
        downsample2 = self.down_sample2(downsample1)
        downsample3 = self.down_sample3(downsample2)

        output = torch.sigmoid(self.output_conv(downsample3))
        return output
