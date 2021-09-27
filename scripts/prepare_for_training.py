import os

from consts import Consts
from src.utils.files_operator import FilesOperator
from src.data_processing.preprocessor import PyWorldPreprocessor


def reset_output_directory():
    FilesOperator.delete_directory(Consts.output_dir_path)
    FilesOperator.create_directory(Consts.output_dir_path)


def reset_cache_directories(cache_dir=Consts.cache_dir, A=Consts.A_dir_name, B=Consts.B_dir_name):
    FilesOperator.delete_directory(cache_dir)
    FilesOperator.create_directory(cache_dir)

    A_output = os.path.join(cache_dir, A)
    B_output = os.path.join(cache_dir, B)

    FilesOperator.create_directory(A_output)
    FilesOperator.create_directory(B_output)


def reset_validation_directories(validation_dir=Consts.validation_output_dir,
                                 A=Consts.A_val_dir_name, B=Consts.B_val_dir_name):
    FilesOperator.delete_directory(validation_dir)
    FilesOperator.create_directory(validation_dir)

    A_output = os.path.join(validation_dir, A)
    B_output = os.path.join(validation_dir, B)

    FilesOperator.create_directory(A_output)
    FilesOperator.create_directory(B_output)


def save_preprocessed_data(data, subcache_dir, cache_dir=Consts.cache_dir):
    mcep_data, f0_data = data
    mcep, mcep_mean, mcep_std = mcep_data
    f0_mean, f0_std = f0_data

    save_path = os.path.join(cache_dir, subcache_dir)
    FilesOperator.save_preprocessed_data(cache_directory=save_path,
                                         spectral_envelope=mcep,
                                         log_f0_mean=f0_mean,
                                         log_f0_std=f0_std,
                                         mcep_mean=mcep_mean,
                                         mcep_std=mcep_std)


def prepare_for_training(data_dir, A_dir, B_dir):
    """
    1. create output directory
    1. reset cache directories
    2. reset validation directories
    3. preprocess data
    4. save preprocessed to cache directory

    :param data_dir: path to directory where training data is located
    :param A_dir: A domain directory name
    :param B_dir: B domain directory name
    """
    reset_output_directory()
    reset_cache_directories()
    reset_validation_directories()

    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_data, B_data = preprocessor.preprocess(data_directory=data_dir,
                                             A_dir=A_dir,
                                             B_dir=B_dir)

    save_preprocessed_data(A_data, subcache_dir=Consts.A_dir_name)
    save_preprocessed_data(B_data, subcache_dir=Consts.B_dir_name)
