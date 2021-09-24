import os
import shutil
import pickle
import librosa
import numpy as np

from src.utils.consts import Consts

ignore_errors = True


class FilesOperator:
    @staticmethod
    def load_signals(data_directory, sampling_rate):
        signals = list()
        for file in os.listdir(data_directory):
            file_path = os.path.join(data_directory, file)
            signal, _ = librosa.load(file_path, sr=sampling_rate, mono=True)
            signals.append(signal)

        return signals

    @staticmethod
    def save_preprocessed_data(cache_directory, spectral_envelope, log_f0_mean, log_f0_std, mcep_mean, mcep_std):
        mcep_file_path = os.path.join(cache_directory, Consts.mcep_norm_file)
        np.savez(mcep_file_path, mean=mcep_mean, std=mcep_std)

        log_f0_file_path = os.path.join(cache_directory, Consts.log_f0_norm_file)
        np.savez(log_f0_file_path, mean=log_f0_mean, std=log_f0_std)

        spectral_envelope_file_path = os.path.join(cache_directory, Consts.spectral_envelope_file)
        with open(spectral_envelope_file_path, 'wb') as file:
            pickle.dump(spectral_envelope, file)

    @staticmethod
    def load_preprocessed_data_normalization_files(cache_directory):
        mcep_file_path = os.path.join(cache_directory, Consts.mcep_norm_file)
        mcep = FilesOperator.load_numpy_npz_file(mcep_file_path)

        log_f0_file_path = os.path.join(cache_directory, Consts.log_f0_norm_file)
        log_f0 = FilesOperator.load_numpy_npz_file(log_f0_file_path)

        return mcep, log_f0

    @staticmethod
    def load_pickle_file(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_numpy_npz_file(file):
        return np.load(file)

    @staticmethod
    def __create_cache_files(directory):
        A_f0_file = os.path.join(directory, Consts.log_f0_norm_file)
        A_mcep_file = os.path.join(directory, Consts.mcep_norm_file)
        A_spectral_envelope_file = os.path.join(directory, Consts.spectral_envelope_file)
        open(A_f0_file, "w")
        open(A_mcep_file, "w")
        open(A_spectral_envelope_file, "w")

    @staticmethod
    def __create_cache_dirs(A_cache_dir, B_cache_dir):
        os.mkdir(Consts.cache_dir)

        os.mkdir(Consts.A_cache_dir)
        os.mkdir(Consts.B_cache_dir)

        FilesOperator.__create_cache_files(A_cache_dir)
        FilesOperator.__create_cache_files(B_cache_dir)

    @staticmethod
    def delete_directory(directory):
        shutil.rmtree(directory, ignore_errors=ignore_errors)

    @staticmethod
    def create_directory(directory):
        os.mkdir(directory)

