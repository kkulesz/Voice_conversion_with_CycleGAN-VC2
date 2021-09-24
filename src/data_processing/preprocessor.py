import os
from abc import ABC
from abc import abstractmethod

from src.data_processing.processing_utils import ProcessingUtils
from src.utils.consts import Consts
from src.utils.files_operator import FilesOperator


class Preprocessor(ABC):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        self._number_of_mceps = number_of_mceps
        self._sampling_rate = sampling_rate
        self._frame_period_in_ms = frame_period_in_ms

    @abstractmethod
    def preprocess(self, data_directory: str, A_dir: str, B_dir: str, cache_directory: str) -> None:
        pass


class PyWorldPreprocessor(Preprocessor):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        super(PyWorldPreprocessor, self).__init__(number_of_mceps, sampling_rate, frame_period_in_ms)

    def preprocess(self, data_directory, A_dir, B_dir, cache_directory):
        A_dataset_dir = os.path.join(data_directory, A_dir)
        B_dataset_dir = os.path.join(data_directory, B_dir)

        self._preprocess_domain(A_dataset_dir, Consts.A_cache_dir)
        self._preprocess_domain(B_dataset_dir, Consts.B_cache_dir)

    def _preprocess_domain(self, data_directory, cache_directory):
        signals = FilesOperator.load_signals(data_directory, self._sampling_rate)

        f0s, _, _, _, mceps = ProcessingUtils \
            .decompose_signals(signals, self._sampling_rate, self._number_of_mceps)

        log_f0_mean, log_f0_std = ProcessingUtils.get_log_f0_mean_and_std(f0s)

        mceps_normalized, mcpes_mean, mceps_std = ProcessingUtils \
            .transpose_and_normalize_mceps(mceps)

        FilesOperator.save_preprocessed_data(cache_directory,
                                             mceps_normalized,
                                             log_f0_mean,
                                             log_f0_std,
                                             mcpes_mean,
                                             mceps_std)


if __name__ == '__main__':
    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_sub_dir, B_sub_dir = Consts.male_to_female
    preprocessor.preprocess(data_directory=Consts.data_dir_vc16,
                            A_dir=A_sub_dir,
                            B_dir=B_sub_dir,
                            cache_directory=Consts.cache_dir)
