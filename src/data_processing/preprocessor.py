import os
from abc import ABC
from abc import abstractmethod

from src.data_processing.processing_utils import ProcessingUtils
from consts import Consts
from src.utils.files_operator import FilesOperator


class Preprocessor(ABC):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        self._number_of_mceps = number_of_mceps
        self._sampling_rate = sampling_rate
        self._frame_period_in_ms = frame_period_in_ms

    @abstractmethod
    def preprocess(self, data_directory: str, A_dir: str, B_dir: str):
        pass


class PyWorldPreprocessor(Preprocessor):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        super(PyWorldPreprocessor, self).__init__(number_of_mceps, sampling_rate, frame_period_in_ms)

    def preprocess(self, data_directory, A_dir, B_dir):
        A_dataset_dir = os.path.join(data_directory, A_dir)
        B_dataset_dir = os.path.join(data_directory, B_dir)

        A_mcep, A_f0 = self._preprocess_domain(A_dataset_dir, A_dir)
        B_mcep, B_f0 = self._preprocess_domain(B_dataset_dir, B_dir)

        return (A_mcep, A_f0), (B_mcep, B_f0)

    def _preprocess_domain(self, data_directory, dir_log_message):
        signals = FilesOperator.load_signals(data_directory, self._sampling_rate)

        f0s, _, _, _, mceps = ProcessingUtils \
            .decompose_signals(signals, self._sampling_rate, self._number_of_mceps)

        log_f0_mean, log_f0_std = ProcessingUtils.get_log_f0_mean_and_std(f0s)

        mceps_normalized, mcpes_mean, mceps_std = ProcessingUtils \
            .transpose_and_normalize_mceps(mceps)

        print(f"- {dir_log_message}: logf0_Mean: {log_f0_mean: .4f}, logf0_Std: {log_f0_std: .4f}")
        # print(f"{dir_log_message}: MCEP_Mean: {mcpes_mean}, MCEP_Std: {mceps_std}")

        return (mceps_normalized, mcpes_mean, mceps_std), (log_f0_mean, log_f0_std)


if __name__ == '__main__':
    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes_cycle_gan_2,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_sub_dir, B_sub_dir = Consts.female_to_male
    preprocessor.preprocess(data_directory=Consts.vc16_training_directory_path,
                            A_dir=A_sub_dir,
                            B_dir=B_sub_dir)
