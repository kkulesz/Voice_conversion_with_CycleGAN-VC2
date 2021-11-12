import os
import torch

from src.data_processing.validator import Validator
from src.model.cycle_gan_vc.generator import Generator
from src.model.cycle_gan_vc.discriminator import Discriminator
from src.utils.files_operator import FilesOperator
from consts import Consts


def print_first_param(model):
    for name, param in model.named_parameters():
        print(f"name: {name}")
        print(f"param: {param}")
        return


if __name__ == '__main__':
    project_dir = Consts.project_dir_path
    cache_dir = os.path.join(project_dir, 'storage\\2021.10.24-25\\cache')

    validator = Validator(A_cache_dir=os.path.join(cache_dir, 'A'),
                          B_cache_dir=os.path.join(cache_dir, 'B'))

    A2B_gen = Generator()
    B2A_gen = Generator()
    A_disc = Discriminator()
    B_disc = Discriminator()

    model_storage_path = os.path.join(project_dir, 'storage\\2021.10.24-25')
    # print_first_param(A2B_gen)
    A2B_gen.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.A2B_generator_file_name))
    # print_first_param(A2B_gen)
    # exit(0)
    B2A_gen.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.B2A_generator_file_name))
    A_disc.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.A_discriminator_file_name))
    B_disc.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.B_discriminator_file_name))

    data_src_dir = os.path.join(project_dir, 'data\\vc-challenge-2016\\vcc2016_training')
    input_file_name = '100020.wav'

    A_input_file_path = os.path.join(data_src_dir, 'SM1', input_file_name)
    B_input_file_path = os.path.join(data_src_dir, 'TF1', input_file_name)

    A_real_numpy, (A_f0, A_ap) = validator.load_and_normalize(A_input_file_path, is_A=True)
    B_real_numpy, (B_f0, B_ap) = validator.load_and_normalize(B_input_file_path, is_A=False)

    A_real_torch_tensor = torch.from_numpy(A_real_numpy).float()
    B_real_torch_tensor = torch.from_numpy(B_real_numpy).float()

    B_fake = A2B_gen(A_real_torch_tensor).detach()
    A_fake = B2A_gen(B_real_torch_tensor).detach()

    A_cycle = B2A_gen(B_fake).detach()
    B_cycle = A2B_gen(A_fake).detach()

    validator.denormalize_and_save(B_fake, A_ap, A_f0, 'B_fake.wav', is_A=True)
    validator.denormalize_and_save(A_fake, B_ap, B_f0, 'A_fake.wav', is_A=False)

    validator.denormalize_and_save(A_cycle, A_ap, A_f0, 'A_cycle.wav', is_A=True)
    validator.denormalize_and_save(B_cycle, B_ap, B_f0, 'B_cycle.wav', is_A=False)
