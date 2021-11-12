import torch
import numpy as np

from src.model.cycle_gan_vc2.generator import GeneratorCycleGan2
from src.model.cycle_gan_vc2.discriminator import DiscriminatorCycleGan2


if __name__ == '__main__':
    # Generator Dimensionality Testing
    num_of_mceps = 36
    np.random.seed(0)
    x = np.random.randn(2, num_of_mceps, 128)
    x = torch.from_numpy(x).float()
    print("Generator input: ", x.shape)
    generator = GeneratorCycleGan2()
    gen_output = generator(x)
    print("Generator output shape: ", gen_output.shape)

    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = DiscriminatorCycleGan2()
    disc_output = discriminator(gen_output)
    print("Discriminator output shape ", disc_output.shape)
