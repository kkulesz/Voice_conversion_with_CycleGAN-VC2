import torch
import torch.nn as nn

from src.model.submodules.glu import GLU
from src.model.submodules.down_sample_1d_layer import DownSample1DLayer
from src.model.submodules.up_sample import UpSampleLayer
from src.model.submodules.residual_block import ResidualBlock


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

        # need to be done in that ugly-repetitive way because:
        #   1. missing copy() implementation
        #   2. if layers are given in list then external `.to(device)` does not work
        self.rb_1 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.rb_2 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.rb_3 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.rb_4 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.rb_5 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.rb_6 = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        self.up_sample_1 = UpSampleLayer(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.up_sample_2 = UpSampleLayer(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.last_conv = nn.Conv1d(in_channels=256, out_channels=24, kernel_size=15, stride=1, padding=7)

    def forward(self, x):
        after_initial_conv = self.initial_conv(x)
        after_first_down_sampling = self.down_sample_1(after_initial_conv)
        after_down_sampling = self.down_sample_2(after_first_down_sampling)

        after_rb_1 = self.rb_1(after_down_sampling)
        after_rb_2 = self.rb_2(after_rb_1)
        after_rb_3 = self.rb_3(after_rb_2)
        after_rb_4 = self.rb_4(after_rb_3)
        after_rb_5 = self.rb_5(after_rb_4)
        after_residual_blocks = self.rb_6(after_rb_5)

        after_first_up_sampling = self.up_sample_1(after_residual_blocks)
        after_up_sampling = self.up_sample_2(after_first_up_sampling)

        result = self.last_conv(after_up_sampling)
        return result
