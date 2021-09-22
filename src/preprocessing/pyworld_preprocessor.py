import os
import numpy as np
import pyworld as pw

from src.consts import Consts
from src.utils.files_operator import FilesOperator
from src.preprocessing.preprocessor import Preprocessor


class PyWorldPreprocessor(Preprocessor):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        super(PyWorldPreprocessor, self).__init__(number_of_mceps, sampling_rate, frame_period_in_ms)

    def preprocess(self, data_directory, A_dir, B_dir, cache_directory):
        A_dataset_dir = os.path.join(data_directory, A_dir)
        B_dataset_dir = os.path.join(data_directory, B_dir)

        self._preprocess_domain(A_dataset_dir, Consts.A_cache_dir)
        self._preprocess_domain(B_dataset_dir, Consts.B_cache_dir)

    def _preprocess_domain(self, data_directory, cache_directory):
        signals = self._load_signals(data_directory)

        f0s, _, _, _, spectral_envelopes = self._decompose_signals(signals)

        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0_mean = log_f0s_concatenated.mean()
        log_f0_std = log_f0s_concatenated.std()

        spectral_envelopes_transposed = self._transpose_in_list(spectral_envelopes)
        spectral_envelopes_normalized, spectral_envelopes_mean, spectral_envelopes_std = \
            self._normalize_spectral_envelope(spectral_envelopes_transposed)

        FilesOperator.save_preprocessed_data(cache_directory,
                                             spectral_envelopes_normalized,
                                             log_f0_mean,
                                             log_f0_std,
                                             spectral_envelopes_mean,
                                             spectral_envelopes_std)

    def _decompose_signals(self, signals):
        f0s = list()
        time_axes = list()
        sps = list()
        aps = list()
        spectral_envelopes = list()

        for signal in signals:
            f0, time, sp, ap = self._decompose_signal(signal)
            sp_envelope = pw.code_spectral_envelope(sp, self._sampling_rate, self._number_of_mceps)  # get MCEPs

            f0s.append(f0)
            time_axes.append(time)
            sps.append(sp)
            aps.append(ap)
            spectral_envelopes.append(sp_envelope)

        return f0s, time_axes, sps, aps, spectral_envelopes

    def _decompose_signal(self, signal):
        signal = signal.astype(np.float64)

        # Note form PyWorld github: If SNR(signal-to-noise-ratio) is low then use 'harvest' instead of 'dio'
        f0, time = pw.dio(signal, self._sampling_rate, f0_floor=Consts.f0_floor, f0_ceil=Consts.f0_ceil)

        # OPTIONAL - TODO: check whether it does not break anything
        # f0 = pw.stonemask(signal, f0, time, self._sampling_rate) # pitch_refinement

        smoothed_spectogram = pw.cheaptrick(signal, f0, time, self._sampling_rate)
        aperiodicity = pw.d4c(signal, f0, time, self._sampling_rate)

        return f0, time, smoothed_spectogram, aperiodicity

    @staticmethod
    def _transpose_in_list(to_transpose):
        transposed_lst = list()
        for array in to_transpose:
            transposed_lst.append(array.T)
        return transposed_lst

    @staticmethod
    def _normalize_spectral_envelope(spectral_envelopes):
        concatenated = np.concatenate(spectral_envelopes, axis=1)
        mean = np.mean(concatenated, axis=1, keepdims=True)
        std = np.std(concatenated, axis=1, keepdims=True)
        normalized = list()
        for single in spectral_envelopes:
            normalized.append((single - mean) / std)

        return normalized, mean, std


if __name__ == '__main__':
    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_sub_dir, B_sub_dir = Consts.male_to_female
    preprocessor.preprocess(data_directory=Consts.data_dir_vc16,
                            A_dir=A_sub_dir,
                            B_dir=B_sub_dir,
                            cache_directory=Consts.cache_dir)