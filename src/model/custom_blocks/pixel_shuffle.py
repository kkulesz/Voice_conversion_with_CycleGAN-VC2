import torch
import torch.nn as nn


class PixelShuffle(nn.Module):
    """
    TODO
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        c_out = x.shape[1] // self.upscale_factor
        w_new = x.shape[2] * self.upscale_factor
        return x.view(n, c_out, w_new)
