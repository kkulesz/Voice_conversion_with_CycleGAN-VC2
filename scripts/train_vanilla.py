import numpy as np

from consts import Consts
from src.model.vanilla_cycle_gan import VanillaCycleGan
from src.utils.files_operator import FilesOperator


def train_vanilla(A_dataset,
                  B_dataset,
                  A_validation_src_directory,
                  B_validation_src_directory,
                  models_storage_directory,
                  load_model,
                  start_from_epoch_number
                  ):
    if load_model:
        load_dir = models_storage_directory
    else:
        load_dir = None

    trainer = VanillaCycleGan(
        A_dataset=A_dataset,
        B_dataset=B_dataset,
        A_validation_source_dir=A_validation_src_directory,
        B_validation_source_dir=B_validation_src_directory,
        A2B_validation_output_dir=Consts.A2B_validation_output_directory_path,
        B2A_validation_output_dir=Consts.B2A_validation_output_directory_path,
        A_cache_dir=Consts.A_cache_directory_path,
        B_cache_dir=Consts.B_cache_directory_path,
        save_models_dir=models_storage_directory,
        load_models_dir=load_dir,
        start_from_epoch_number=start_from_epoch_number
    )

    trainer.train()


if __name__ == '__main__':
    A_dir = Consts.current_A_training_dir
    B_dir = Consts.current_B_training_dir
    A_validation_source_dir = Consts.current_A_val_dir
    B_validation_source_dir = Consts.current_B_val_dir
    print(f"TRAINING FROM: {A_dir} TO: {B_dir}")
    print(f"VALIDATING FROM: {A_validation_source_dir} TO: {B_validation_source_dir}")
    ###########################################################
    models_storage_dir = Consts.models_storage_directory_path
    load_model = False
    start_from_epoch_number = 0
    ###########################################################

    A_dataset = FilesOperator.load_pickle_file(Consts.A_preprocessed_dataset_file_path)
    B_dataset = FilesOperator.load_pickle_file(Consts.B_preprocessed_dataset_file_path)
    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        train_vanilla(A_dataset,
                      B_dataset,
                      A_validation_source_dir,
                      B_validation_source_dir,
                      models_storage_dir,
                      load_model=load_model,
                      start_from_epoch_number=start_from_epoch_number)
