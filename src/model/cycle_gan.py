import torch
import numpy as np

from src.model.generator import Generator
from src.model.discriminator import Discriminator

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    np.random.seed(0)
    x = np.random.randn(158, 24, 128)
    x = torch.from_numpy(x).float()

    generator_output = generator(x)
    discriminator_output = discriminator(generator_output)
