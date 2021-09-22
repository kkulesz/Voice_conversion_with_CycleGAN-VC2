import os
import shutil
import pickle
import numpy as np

from src.consts import Consts


class FilesOperator:
    @staticmethod
    def reset_cache_dirs():
        FilesOperator.__delete_cache_dirs()
        FilesOperator.__create_cache_dirs()

    @staticmethod
    def save_preprocessed_data(cache_directory, spectral_envelope, log_f0_mean, log_f0_std, mcep_mean, mcep_std):
        mcep_file = os.path.join(cache_directory, Consts.mcep_norm_file)
        np.savez(mcep_file, mean=mcep_mean, std=mcep_std)

        log_f0_file = os.path.join(cache_directory, Consts.log_f0_norm_file)
        np.savez(log_f0_file, mean=log_f0_mean, std=log_f0_std)

        spectral_envelope_file = os.path.join(cache_directory, Consts.spectral_envelope_file)
        with open(spectral_envelope_file, 'wb') as file:
            pickle.dump(spectral_envelope, file)

    @staticmethod
    def load_pickle_file(file):
        """
        TODO:
        """
        pass

    @staticmethod
    def load_npz_file(file):
        """
        TODO:
        """
        pass

    @staticmethod
    def __create_cache_files(directory):
        A_f0_file = os.path.join(directory, Consts.log_f0_norm_file)
        A_mcep_file = os.path.join(directory, Consts.mcep_norm_file)
        A_spectral_envelope_file = os.path.join(directory, Consts.spectral_envelope_file)
        open(A_f0_file, "w")
        open(A_mcep_file, "w")
        open(A_spectral_envelope_file, "w")

    @staticmethod
    def __create_cache_dirs():
        os.mkdir(Consts.cache_dir)

        os.mkdir(Consts.A_cache_dir)
        os.mkdir(Consts.B_cache_dir)

        FilesOperator.__create_cache_files(Consts.A_cache_dir)
        FilesOperator.__create_cache_files(Consts.B_cache_dir)

    @staticmethod
    def __delete_cache_dirs():
        shutil.rmtree(Consts.cache_dir)
