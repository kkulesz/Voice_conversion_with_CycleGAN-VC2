import torch
from torch.utils.data import DataLoader

from src.consts import Consts
from src.utils.files_operator import FilesOperator
from src.preprocessing.pyworld_preprocessor import PyWorldPreprocessor
from src.dataset.preprocessed_dataset import PreprocessedDataset
from src.modules.generator import Generator
from src.modules.discriminator import Discriminator


def print_tensor_info(source: str, x: torch.Tensor) -> None:
    print(f"{source}:\n"
          f"\tshape={x.shape}\n"
          f"\ttype={x.dtype}\n")


if __name__ == '__main__':
    FilesOperator.reset_cache_dirs()

    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_dir, B_dir = Consts.male_to_female
    preprocessor.preprocess(data_directory=Consts.data_dir_vc16,
                            A_dir=A_dir,
                            B_dir=B_dir,
                            cache_directory=Consts.cache_dir)

    dataset = PreprocessedDataset(A_dataset_file=Consts.A_preprocessed_dataset_file,
                                  B_dataset_file=Consts.B_preprocessed_dataset_file,
                                  number_of_frames=Consts.number_of_frames)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=Consts.mini_batch_size,
                            shuffle=False,
                            drop_last=False)

    generator = Generator()
    discriminator = Discriminator()
    for i, (real_A, real_B) in enumerate(dataloader):
        print_tensor_info('Input', real_A)

        after_generator = generator(real_A)
        print_tensor_info('Generator output', after_generator)

        after_discriminator = discriminator(after_generator)
        print_tensor_info('Discriminator output', after_discriminator)

        # TODO: make unit tests later
        assert real_A.shape == after_generator.shape
        print(after_discriminator[0][0][0])

        exit(0)  # do not iterate, one is enough
