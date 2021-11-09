import pyworld as pw
import numpy as np

from consts import Consts


class ProcessingUtils:
    @staticmethod
    def decompose_signals(signals, sampling_rate, number_of_mceps):
        f0s = list()
        time_axes = list()
        sps = list()
        aps = list()
        mceps = list()

        for signal in signals:
            f0, time, sp, ap = ProcessingUtils.decompose_signal(signal, sampling_rate)
            mcep = pw.code_spectral_envelope(sp, sampling_rate, number_of_mceps)

            f0s.append(f0)
            time_axes.append(time)
            sps.append(sp)
            aps.append(ap)
            mceps.append(mcep)

        return f0s, time_axes, sps, aps, mceps

    @staticmethod
    def decompose_signal(signal, sampling_rate):
        signal = signal.astype(np.float64)

        # Note form PyWorld github: If SNR(signal-to-noise-ratio) is low then use 'harvest' instead of 'dio'
        f0, time = pw.harvest(signal, sampling_rate, frame_period=Consts.frame_period_in_ms, f0_floor=Consts.f0_floor, f0_ceil=Consts.f0_ceil)

        # OPTIONAL - TODO: check whether it does not break anything
        # f0 = pw.stonemask(signal, f0, time, self._sampling_rate) # pitch_refinement

        smoothed_spectogram = pw.cheaptrick(signal, f0, time, sampling_rate)
        aperiodicity = pw.d4c(signal, f0, time, sampling_rate)

        return f0, time, smoothed_spectogram, aperiodicity

    @staticmethod
    def get_log_f0_mean_and_std(f0s):
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0_mean = log_f0s_concatenated.mean()
        log_f0_std = log_f0s_concatenated.std()

        return log_f0_mean, log_f0_std

    @staticmethod
    def transpose_and_normalize_mceps(mceps):
        transposed_mceps = ProcessingUtils._transpose_in_list(mceps)
        concatenated = np.concatenate(transposed_mceps, axis=1)
        mean = np.mean(concatenated, axis=1, keepdims=True)
        std = np.std(concatenated, axis=1, keepdims=True)
        normalized = list()
        for mcep in transposed_mceps:
            normalized_mcep = ProcessingUtils.normalize_mcep(mcep, mean, std)
            normalized.append(normalized_mcep)

        return normalized, mean, std

    @staticmethod
    def normalize_mcep(mcep, mean, std):
        return (mcep - mean) / std

    @staticmethod
    def denormalize_mcep(mcep, mean, std):
        return (mcep * std) + mean

    @staticmethod
    def gaussian_normalization(log_f0, mean_source, std_source, mean_target, std_target):
        return np.exp((log_f0 - mean_source) / std_source * std_target + mean_target)

    @staticmethod
    def pad_signal(signal, sr, frame_period, multiple):
        assert signal.ndim == 1  # TODO get rid of it
        number_of_frames = len(signal)

        how_many_after_pad = ProcessingUtils.count_padding(number_of_frames, sr, frame_period, multiple)
        difference = how_many_after_pad - number_of_frames
        pad_left = difference // 2
        pad_right = difference - pad_left

        padded = np.pad(signal, (pad_left, pad_right), 'constant', constant_values=0)
        return padded

    @staticmethod
    def count_padding(number_of_frames, sampling_rate, frame_period, multiple):
        parameter = (sampling_rate * frame_period / 1000)
        padding_number = (np.ceil((np.floor(number_of_frames / parameter) + 1) / multiple + 1) * multiple - 1) * parameter
        return int(padding_number)

    @staticmethod
    def _transpose_in_list(to_transpose):
        transposed_lst = list()
        for array in to_transpose:
            transposed_lst.append(array.T)
        return transposed_lst
