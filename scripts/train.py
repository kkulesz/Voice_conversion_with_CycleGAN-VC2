import os
import numpy as np

from consts import Consts

from src.training.train_class import CycleGanTraining

from scripts.prepare_for_training import prepare_for_training
from scripts.download import download_vc2016_dataset


def prepare_training_class(A_validation_src_directory, B_validation_src_directory):
    return CycleGanTraining(
        A_data_file=Consts.A_preprocessed_dataset_file_path,
        B_data_file=Consts.B_preprocessed_dataset_file_path,
        number_of_frames=Consts.number_of_frames,
        batch_size=Consts.mini_batch_size,
        A_validation_dir=A_validation_src_directory,
        B_validation_dir=B_validation_src_directory,
        A_output_dir=Consts.A_validation_output_directory_path,
        B_output_dir=Consts.B_validation_output_directory_path,
        A_cache_dir=Consts.A_cache_directory_path,
        B_cache_dir=Consts.B_cache_directory_path
    )


if __name__ == '__main__':
    # What to train
    A_dir, B_dir = Consts.male_to_female

    download_destination = Consts.vc16_data_directory_path
    training_data_dir = Consts.vc16_training_directory_path
    validation_data_dir = Consts.vc16_validation_directory_path

    A_validation_source_dir = os.path.join(validation_data_dir, A_dir)
    B_validation_source_dir = os.path.join(validation_data_dir, B_dir)

    download_vc2016_dataset(download_destination)
    prepare_for_training(training_data_dir, A_dir, B_dir)

    cycleGanTraining = prepare_training_class(A_validation_source_dir, B_validation_source_dir)
    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        cycleGanTraining.train()
