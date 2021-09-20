import os
import numpy as np
import pyworld as pw
import pickle

from src.consts import Consts
from src.preprocessing.preprocessor import Preprocessor


class PyWorldPreprocessor(Preprocessor):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        super(PyWorldPreprocessor, self).__init__(number_of_mceps, sampling_rate, frame_period_in_ms)

    def preprocess(self, data_directory, cache_directory):
        A, B = Consts.male_to_female
        A_dataset_dir = os.path.join(data_directory, A)
        B_dataset_dir = os.path.join(data_directory, B)

        self._preprocess_domain(A_dataset_dir, Consts.A_cache_dir)
        self._preprocess_domain(B_dataset_dir, Consts.B_cache_dir)

    def _preprocess_domain(self, data_directory, cache_directory):
        signals = self._load_signals(data_directory)

        f0s, time_axes, _, _, spectral_envelopes = self._decompose_signals(signals)

        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0_mean = log_f0s_concatenated.mean()
        log_f0_std = log_f0s_concatenated.std()

        spectral_envelopes_transposed = self._transpose_in_list(spectral_envelopes)
        spectral_envelopes_normalized, spectral_envelopes_mean, spectral_envelopes_std = \
            self._normalize_spectral_envelope(spectral_envelopes_transposed)

        self._save_preprocessed_data(cache_directory,
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

        # Note form PyWorld github: Is SNR(signal-to-noise-ratio) is low then use 'harvest' instead of 'dio'
        f0, time = pw.dio(signal, self._sampling_rate, f0_floor=Consts.f0_floor, f0_ceil=Consts.f0_ceil)
        # f0 = pw.stonemask(signal, f0, time, self._sampling_rate) # pitch_refinement - OPTIONAL - TODO: check whether it does not break anything
        smoothed_spectogram = pw.cheaptrick(signal, f0, time, self._sampling_rate)
        aperiodicity = pw.d4c(signal, f0, time, self._sampling_rate)

        return f0, time, smoothed_spectogram, aperiodicity

    def _transpose_in_list(self, to_transpose):
        transposed_lst = list()
        for array in to_transpose:
            transposed_lst.append(array.T)
        return transposed_lst

    def _normalize_spectral_envelope(self, spectral_envelopes):
        concatenated = np.concatenate(spectral_envelopes, axis=1)
        mean = np.mean(concatenated, axis=1, keepdims=True)
        std = np.std(concatenated, axis=1, keepdims=True)
        normalized = list()
        for single in spectral_envelopes:
            normalized.append((single - mean) / std)

        return normalized, mean, std

    def _save_preprocessed_data(self, cache_directory, spectral_envelope, log_f0_mean, log_f0_std, mcep_mean, mcep_std):
        mcep_file = os.path.join(cache_directory, Consts.mcep_file)
        np.savez(mcep_file, mean=mcep_mean, std=mcep_std)

        log_f0_file = os.path.join(cache_directory, Consts.f0_file)
        np.savez(log_f0_file, mean=log_f0_mean, std=log_f0_std)

        spectral_envelope_file = os.path.join(cache_directory, Consts.spectral_envelope_file)
        with open(spectral_envelope_file, 'wb') as file:
            pickle.dump(spectral_envelope, file)


if __name__ == '__main__':
    preprocessor = PyWorldPreprocessor(number_of_mceps=24,
                                       sampling_rate=16000,
                                       frame_period_in_ms=5.0)

    preprocessor.preprocess(Consts.vc16_data_dir, Consts.cache_dir)
