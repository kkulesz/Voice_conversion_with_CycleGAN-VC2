import torch.nn as nn

from src.model.custom_blocks.glu import GLU
from src.model.custom_blocks.down_sample import DownSample
from src.model.custom_blocks.residual_block import ResidualBlock


# TODO: find out what padding is for and decide its valuep
#  for now it's copied from pytorch followup work but i dont know where (s)he got it from
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=128, kernel_size=(1, 15), stride=(1, 1), padding=7),
            GLU()
        )

        self.down_sample_1 = DownSample(in_channels=128, out_channels=256, kernel_size=(1, 5), stride=(1, 2), padding=1)
        self.down_sample_2 = DownSample(in_channels=256, out_channels=512, kernel_size=(1, 5), stride=(1, 2), padding=2)

        self.residual_block = ResidualBlock(in_channels=512, out_channels=1024, kernel_size=(1, 3), padding=1)
        self.number_of_residual_block_iteration = 6

        #  TODO: upsample
        #  TODO: last conv layer

    def forward(self, x):
        after_initial_conv = self.initial_conv(x)
        after_first_down_sampling = self.down_sample_1(after_initial_conv)
        after_down_sampling = self.down_sample_1(after_first_down_sampling)

        for_residual_block = after_down_sampling
        for i in range(self.number_of_residual_block_iteration):
            for_residual_block = self.residual_block(for_residual_block)

        return 0
