import os

from src.consts import Consts
from src.utils.files_operator import FilesOperator

from src.preprocessing.pyworld_preprocessor import PyWorldPreprocessor
from src.training.train_class import CycleGanTraining


def prepare_fo_training(data_dir, A_dir, B_dir):
    FilesOperator.reset_cache_dirs()
    FilesOperator.reset_validation_dirs()

    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    preprocessor.preprocess(data_directory=data_dir,
                            A_dir=A_dir,
                            B_dir=B_dir,
                            cache_directory=Consts.cache_dir)


if __name__ == '__main__':
    A_dir, B_dir = Consts.male_to_female
    data_dir = Consts.data_dir_vc16

    # TODO: make it more convenient than comment-uncomment
    # prepare_fo_training(data_dir, A_dir, B_dir)

    A_validation_source_dir = os.path.join(data_dir, A_dir)
    B_validation_source_dir = os.path.join(data_dir, B_dir)

    cycleGanTraining = CycleGanTraining(
        A_data_file=Consts.A_preprocessed_dataset_file,
        B_data_file=Consts.B_preprocessed_dataset_file,
        number_of_frames=Consts.number_of_frames,
        batch_size=Consts.mini_batch_size,
        A_validation_dir=A_validation_source_dir,
        B_validation_dir=B_validation_source_dir,
        A_output_dir=Consts.A_output_dir,
        B_output_dir=Consts.B_output_dir,
        A_cache_dir=Consts.A_cache_dir,
        B_cache_dir=Consts.B_cache_dir
    )

    cycleGanTraining.train()
