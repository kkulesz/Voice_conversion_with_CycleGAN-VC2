import os
import torch

from src.data_processing.validator import Validator
from src.utils.files_operator import FilesOperator
from consts import Consts

# from src.model.cycle_gan_vc.generator import Generator
# from src.model.cycle_gan_vc.discriminator import Discriminator

from src.model.cycle_gan_vc2.generator import GeneratorCycleGan2
from src.model.cycle_gan_vc2.discriminator import DiscriminatorCycleGan2


def print_first_param(model):
    for name, param in model.named_parameters():
        print(f"name: {name}")
        print(f"param: {param}")
        return


if __name__ == '__main__':
    ######################################################################################################
    trained_models_storage_path = 'E:\\STUDIA\inzynierka\\2_moje_przygotowania\\4.trenowanie\\2.SM1-TM1'
    val_dir = 'E:\\STUDIA\inzynierka\\2_moje_przygotowania\\3.kod\\moje_repo\\data\\vc-challenge-2016\\evaluation_all'
    A_val_src_dir = os.path.join(val_dir, 'SM1')
    B_val_src_dir = os.path.join(val_dir, 'TM1')
    A_input_file_path = os.path.join(A_val_src_dir, '200005.wav')
    B_input_file_path = os.path.join(B_val_src_dir, '200005.wav')
    ######################################################################################################

    model_storage_path = os.path.join(trained_models_storage_path, 'models_storage')
    cache_dir = os.path.join(trained_models_storage_path, 'CACHE')
    validator = Validator(A_cache_dir=os.path.join(cache_dir, 'A'), B_cache_dir=os.path.join(cache_dir, 'B'))

    # A2B_gen = Generator()
    # B2A_gen = Generator()
    # A_disc = Discriminator()
    # B_disc = Discriminator()
    A2B_gen = GeneratorCycleGan2()
    B2A_gen = GeneratorCycleGan2()
    A_disc = DiscriminatorCycleGan2()
    B_disc = DiscriminatorCycleGan2()
    A2B_gen.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.A2B_generator_file_name))
    B2A_gen.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.B2A_generator_file_name))
    # A_disc.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.A_discriminator_file_name))
    # B_disc.load_state_dict(FilesOperator.load_model(model_storage_path, Consts.B_discriminator_file_name))

    A_real, (A_f0, A_ap) = validator.load_and_normalize(A_input_file_path, is_A=True)
    B_real, (B_f0, B_ap) = validator.load_and_normalize(B_input_file_path, is_A=False)

    B_fake = A2B_gen(A_real.float()).detach()
    A_fake = B2A_gen(B_real.float()).detach()
    validator.denormalize_and_save(B_fake, A_ap, A_f0, 'B_fake.wav', is_A=False)
    validator.denormalize_and_save(A_fake, B_ap, B_f0, 'A_fake.wav', is_A=True)

    A_cycle = B2A_gen(B_fake).detach()
    B_cycle = A2B_gen(A_fake).detach()
    validator.denormalize_and_save(A_cycle, A_ap, A_f0, 'A_cycle.wav', is_A=True)
    validator.denormalize_and_save(B_cycle, B_ap, B_f0, 'B_cycle.wav', is_A=False)
