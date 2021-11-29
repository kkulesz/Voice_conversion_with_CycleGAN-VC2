import os

import pytorch_lightning as pl

from consts import Consts
from src.model.lightning_cycle_gan import LightningCycleGan
from src.utils.files_operator import FilesOperator


def train_lightning(
        A_dataset,
        B_dataset,
        A_validation_source_dir,
        B_validation_source_dir,
        models_storage_directory,
        load_model=False,
        start_from_epoch_number=0
):
    trainer = pl.Trainer(fast_dev_run=False,
                         gpus=1,
                         check_val_every_n_epoch=2,  # Consts.dump_validation_file_epoch_frequency,
                         enable_progress_bar=False
                         )
    cycle_gan = LightningCycleGan(
        A_dataset=A_dataset,
        B_dataset=B_dataset,
        A_validation_source_dir=A_validation_source_dir,
        B_validation_source_dir=B_validation_source_dir,
        A2B_validation_output_dir=Consts.A2B_validation_output_directory_path,
        B2A_validation_output_dir=Consts.B2A_validation_output_directory_path,
        A_cache_dir=Consts.A_cache_directory_path,
        B_cache_dir=Consts.B_cache_directory_path
    )

    trainer.fit(cycle_gan)


if __name__ == '__main__':
    A_dir = Consts.current_A_training_dir
    B_dir = Consts.current_B_training_dir
    A_validation_source_dir = Consts.current_A_val_dir
    B_validation_source_dir = Consts.current_B_val_dir
    print(f"FROM: {A_dir} TO: {B_dir}")

    models_storage_dir = Consts.models_storage_directory_path

    A_dataset = FilesOperator.load_pickle_file(Consts.A_preprocessed_dataset_file_path)
    B_dataset = FilesOperator.load_pickle_file(Consts.B_preprocessed_dataset_file_path)



    train_lightning(A_dataset,
                    B_dataset,
                    A_validation_source_dir,
                    B_validation_source_dir,
                    models_storage_dir,
                    load_model=False,
                    start_from_epoch_number=0)
