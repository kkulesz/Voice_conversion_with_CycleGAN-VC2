import os
import shutil
import pickle
import librosa
import torch
import numpy as np
import pandas as pd

from consts import Consts

# errors behaviour
ignore_errors = True
exist_ok = True


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
        mcep_file_path = os.path.join(cache_directory, Consts.mcep_norm_filename)
        np.savez(mcep_file_path, mean=mcep_mean, std=mcep_std)

        log_f0_file_path = os.path.join(cache_directory, Consts.log_f0_norm_filename)
        np.savez(log_f0_file_path, mean=log_f0_mean, std=log_f0_std)

        spectral_envelope_file_path = os.path.join(cache_directory, Consts.spectral_envelope_filename)
        with open(spectral_envelope_file_path, 'wb') as file:
            pickle.dump(spectral_envelope, file)

    @staticmethod
    def load_preprocessed_data_normalization_files(cache_directory):
        mcep_file_path = os.path.join(cache_directory, Consts.mcep_norm_filename)
        mcep = FilesOperator.load_numpy_npz_file(mcep_file_path)

        log_f0_file_path = os.path.join(cache_directory, Consts.log_f0_norm_filename)
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
    def save_model(model, save_directory, storage_file_name):
        storage_path = os.path.join(save_directory, storage_file_name)
        torch.save(model.state_dict(), storage_path)

    @staticmethod
    def load_model(load_directory, storage_file_name):
        storage_path = os.path.join(load_directory, storage_file_name)
        return torch.load(storage_path)

    @staticmethod
    def save_list(loss_array, save_directory, file_name):
        storage_path = os.path.join(save_directory, file_name)
        pd.DataFrame(loss_array).to_csv(storage_path)

    @staticmethod
    def delete_directory(directory):
        shutil.rmtree(directory, ignore_errors=ignore_errors)

    @staticmethod
    def create_directory(directory):
        os.makedirs(directory, exist_ok=exist_ok)

