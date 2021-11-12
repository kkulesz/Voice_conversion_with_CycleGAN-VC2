import os

from consts import Consts
from src.utils.files_operator import FilesOperator
from src.data_processing.preprocessor import PyWorldPreprocessor


def _make_sure_output_dir_exist():
    if not os.path.exists(Consts.output_dir_path):
        FilesOperator.create_directory(Consts.output_dir_path)


def _reset_cache_directories(cache_dir=Consts.cache_directory_path, A=Consts.A_dir_name, B=Consts.B_dir_name):
    FilesOperator.delete_directory(cache_dir)
    FilesOperator.create_directory(cache_dir)

    A_output = os.path.join(cache_dir, A)
    B_output = os.path.join(cache_dir, B)

    FilesOperator.create_directory(A_output)
    FilesOperator.create_directory(B_output)


def _reset_validation_directories(validation_dir=Consts.validation_output_directory,
                                  A=Consts.A2B_validation_output_directory_name,
                                  B=Consts.B2A_validation_output_directory_name):
    FilesOperator.delete_directory(validation_dir)
    FilesOperator.create_directory(validation_dir)

    A_output = os.path.join(validation_dir, A)
    B_output = os.path.join(validation_dir, B)

    FilesOperator.create_directory(A_output)
    FilesOperator.create_directory(B_output)


def _save_preprocessed_data(data, subcache_dir, cache_dir=Consts.cache_directory_path):
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


def _make_sure_storage_dir_exist(storage_dir):
    if not os.path.exists(storage_dir):
        FilesOperator.create_directory(storage_dir)


def prepare_for_training(data_dir, A_dir, B_dir, storage_dir):
    """
    1. create output directory IF IT DOES NOT EXIST
    2. reset cache directories
    3. reset validation directories
    4. create storage directory IF IT DOES NOT EXIST
    4. preprocess data
    5. save preprocessed to cache directory

    :param data_dir: path to directory where training data is located
    :param A_dir: A domain directory name
    :param B_dir: B domain directory name
    :param storage_dir: path to directory to  which models and their losses are saved
    """
    _make_sure_output_dir_exist()
    _reset_cache_directories()
    _reset_validation_directories()
    _make_sure_storage_dir_exist(storage_dir)

    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mceps,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_data, B_data = preprocessor.preprocess(data_directory=data_dir,
                                             A_dir=A_dir,
                                             B_dir=B_dir)

    _save_preprocessed_data(A_data, subcache_dir=Consts.A_dir_name)
    _save_preprocessed_data(B_data, subcache_dir=Consts.B_dir_name)
