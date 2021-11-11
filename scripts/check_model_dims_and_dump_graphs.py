import os
import torch

from src.model.generator import Generator
from src.model.discriminator import Discriminator

from consts import Consts
from src.utils.files_operator import FilesOperator

"""
1. checking models dimensionality
2. dumping graphs of both models in storage dir 
"""
if __name__ == '__main__':
    generator_output_file = os.path.join(Consts.my_storage, "generator_graph")
    discriminator_output_file = os.path.join(Consts.my_storage, "discriminator_graph")

    generator = Generator()
    discriminator = Discriminator()

    x = torch.randn(158, 24, 128)
    generated = generator(x)
    discriminated_xd = discriminator(generated.detach())

    FilesOperator.dump_torchviz_graph(generated,
                                      params=dict(list(generator.named_parameters())),
                                      file_name=generator_output_file)

    FilesOperator.dump_torchviz_graph(discriminated_xd,
                                      params=dict(list(discriminator.named_parameters())),
                                      file_name=discriminator_output_file)
