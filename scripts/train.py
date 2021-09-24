import os
import numpy as np

from src.utils.consts import Consts

from src.training.train_class import CycleGanTraining

from scripts.prepare_for_training import prepare_for_training


def prepare_training_class(A_validation_src_directory, B_validation_src_directory):
    return CycleGanTraining(
        A_data_file=Consts.A_preprocessed_dataset_file,
        B_data_file=Consts.B_preprocessed_dataset_file,
        number_of_frames=Consts.number_of_frames,
        batch_size=Consts.mini_batch_size,
        A_validation_dir=A_validation_src_directory,
        B_validation_dir=B_validation_src_directory,
        A_output_dir=Consts.A_output_dir,
        B_output_dir=Consts.B_output_dir,
        A_cache_dir=Consts.A_cache_dir,
        B_cache_dir=Consts.B_cache_dir
    )


if __name__ == '__main__':
    # TODO: download data script
    A_dir, B_dir = Consts.male_to_female
    data_dir = Consts.data_dir_vc16

    A_validation_source_dir = os.path.join(data_dir, A_dir)
    B_validation_source_dir = os.path.join(data_dir, B_dir)

    prepare_for_training(data_dir, A_dir, B_dir)
    cycleGanTraining = prepare_training_class(A_validation_source_dir, B_validation_source_dir)

    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        cycleGanTraining.train()
