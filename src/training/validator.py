import pyworld as pw
import numpy as np

from src.utils.files_operator import FilesOperator


class Validator:
    def __init__(self, A_cache_dir, B_cache_dir):
        (A_mcep, A_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(A_cache_dir)
        (B_mcep, B_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(B_cache_dir)

        self.A_mcep_mean, self.A_mcep_std, self.A_log_f0_mean, self.A_log_f0_std = \
            Validator._unpack_normalizations(A_mcep, A_log_f0)
        self.B_mcep_mean, self.B_mcep_std, self.B_log_f0_mean, self.B_log_f0_std = \
            Validator._unpack_normalizations(B_mcep, B_log_f0)

    def load_and_normalize(self, file_path):
        pass

    def denormalize_and_save(self, save_directory, file_name):
        pass

    @staticmethod
    def _unpack_normalizations(mcep, log_f0):
        mcep_mean = mcep['mean']
        mcep_std = mcep['std']

        log_f0_mean = log_f0['mean']
        log_f0_std = log_f0['std']

        return mcep_mean, mcep_std, log_f0_mean, log_f0_std
