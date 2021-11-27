import os
import numpy as np
import wandb

from consts import Consts

from scripts.download import download_vc2016_dataset
from scripts.prepare_directories_and_preprocess import prepare_directories_and_preprocess
from scripts.train_vanilla import train_vanilla
from scripts.train_lightning import train_lightning
from src.utils.files_operator import FilesOperator

if __name__ == '__main__':
    # ==========================================================
    A_dir, B_dir = Consts.male_to_male
    print(f"FROM: {A_dir} TO: {B_dir}")

    download_destination = Consts.vc16_data_directory_path
    training_data_dir = Consts.vc16_training_directory_path
    validation_data_dir = Consts.vc16_validation_directory_path

    # ==========================================================
    models_storage_dir = Consts.models_storage_directory_path
    load_model = False
    start_from_epoch_number = 0
    # ==========================================================
    is_vanilla = True
    # ==========================================================

    A_validation_source_dir = os.path.join(validation_data_dir, A_dir)
    B_validation_source_dir = os.path.join(validation_data_dir, B_dir)

    # print("Downloading...")
    # download_vc2016_dataset(download_destination)
    #
    print("Preprocessing...")
    prepare_directories_and_preprocess(training_data_dir, A_dir, B_dir, models_storage_dir)

    A_dataset = FilesOperator.load_pickle_file(Consts.A_preprocessed_dataset_file_path)
    B_dataset = FilesOperator.load_pickle_file(Consts.B_preprocessed_dataset_file_path)

    print("Initializing and training...")
    wandb.login()
    wandb.init(project='cycleGan-test-run')
    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        if is_vanilla:
            train_vanilla(
                A_dataset,
                B_dataset,
                A_validation_source_dir,
                B_validation_source_dir,
                models_storage_dir,
                load_model=load_model,
                start_from_epoch_number=start_from_epoch_number)
        else:
            train_lightning(
                A_dataset,
                B_dataset,
                A_validation_source_dir,
                B_validation_source_dir,
                models_storage_dir,
                load_model,
                start_from_epoch_number=start_from_epoch_number)
