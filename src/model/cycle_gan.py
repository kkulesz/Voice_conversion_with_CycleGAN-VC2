import torch
import torch.nn as nn

from src.model.generator import Generator
from src.model.discriminator import Discriminator


# class CycleGan(nn.Module):
#     def __init__(self, generator, discriminator):
#         super(CycleGan, self).__init__()
#         self.generator = generator
#         self.discriminator = discriminator


if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    x = torch.randn(158, 24, 128)

    generator_output = generator(x)
    discriminator_output = discriminator(generator_output)
