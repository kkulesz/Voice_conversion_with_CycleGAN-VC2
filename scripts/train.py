import os
import numpy as np

from consts import Consts

from src.training.train_class import CycleGanTraining

from scripts.prepare_for_training import prepare_for_training
from scripts.download import download_vc2016_dataset


def prepare_training_class(A_validation_src_directory,
                           B_validation_src_directory,
                           models_storage_directory, load_model=False):

    if load_model:
        load_dir = models_storage_directory
    else:
        load_dir = None

    return CycleGanTraining(
        A_data_file=Consts.A_preprocessed_dataset_file_path,
        B_data_file=Consts.B_preprocessed_dataset_file_path,
        A_validation_source_dir=A_validation_src_directory,
        B_validation_source_dir=B_validation_src_directory,
        A2B_validation_output_dir=Consts.A2B_validation_output_directory_path,
        B2A_validation_output_dir=Consts.B2A_validation_output_directory_path,
        A_cache_dir=Consts.A_cache_directory_path,
        B_cache_dir=Consts.B_cache_directory_path,
        save_models_dir=models_storage_directory,
        load_models_dir=load_dir
    )


if __name__ == '__main__':
    # ==========================================================
    A_dir, B_dir = Consts.female_to_male
    print(f"FROM: {A_dir} TO: {B_dir}")

    download_destination = Consts.vc16_data_directory_path
    training_data_dir = Consts.vc16_training_directory_path
    validation_data_dir = Consts.vc16_validation_directory_path

    models_storage_dir = Consts.models_storage_directory_path
    load_model = False
    # ==========================================================

    A_validation_source_dir = os.path.join(validation_data_dir, A_dir)
    B_validation_source_dir = os.path.join(validation_data_dir, B_dir)

    print("Downloading...")
    download_vc2016_dataset(download_destination)

    print("Preprocessing...")
    prepare_for_training(training_data_dir, A_dir, B_dir, models_storage_dir)

    print("Initializing...")
    cycleGanTraining = prepare_training_class(A_validation_source_dir,
                                              B_validation_source_dir,
                                              models_storage_dir, load_model=load_model)
    print("Starting training...")
    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        cycleGanTraining.train()
