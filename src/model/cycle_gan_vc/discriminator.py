import torch
import torch.nn as nn

from src.model.cycle_gan_vc.submodules.glu import GLU
from src.model.cycle_gan_vc.submodules.down_sample_2d_layer import DownSample2DLayer


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=[3, 4], stride=[1, 2], padding=[1, 1]),
            GLU()
        )

        self.down_sample_1 = \
            DownSample2DLayer(in_channels=128, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=1)
        self.down_sample_2 = \
            DownSample2DLayer(in_channels=256, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=1)
        self.down_sample_3 = \
            DownSample2DLayer(in_channels=512, out_channels=1024, kernel_size=[5, 4], stride=[1, 2], padding=[2, 1])

        self.fully_connected_layer = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x: torch.Tensor):
        unsqueezed = x.unsqueeze(1)  # (batch_size, 1, number_of_features, time)
        after_initial_conv = self.initial_conv(unsqueezed)
        after_down_sampling_1 = self.down_sample_1(after_initial_conv)
        after_down_sampling_2 = self.down_sample_2(after_down_sampling_1)
        after_down_sampling_3 = self.down_sample_3(after_down_sampling_2)
        after_down_sampling = after_down_sampling_3.contiguous().permute(0, 2, 3, 1).contiguous()
        after_fully_connected_layer = self.fully_connected_layer(after_down_sampling)
        result = torch.sigmoid(after_fully_connected_layer)

        return result

