import librosa
import pyworld as pw
import numpy as np
import soundfile as sf

from src.consts import Consts
from src.utils.files_operator import FilesOperator
from src.preprocessing.pyworld_preprocessor import PyWorldPreprocessor


class Validator:
    def __init__(self, A_cache_dir, B_cache_dir):
        (A_mcep, A_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(A_cache_dir)
        (B_mcep, B_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(B_cache_dir)

        self.A_mcep_mean, self.A_mcep_std, self.A_log_f0_mean, self.A_log_f0_std = \
            Validator._unpack_normalizations(A_mcep, A_log_f0)
        self.B_mcep_mean, self.B_mcep_std, self.B_log_f0_mean, self.B_log_f0_std = \
            Validator._unpack_normalizations(B_mcep, B_log_f0)

    def load_and_normalize(self, file_path: str, is_A: bool):
        signal, _ = librosa.load(file_path, sr=Consts.sampling_rate, mono=True)
        padded = Validator._pad_signal(signal,
                                       sr=Consts.sampling_rate,
                                       frame_period=Consts.frame_period_in_ms,
                                       multiple=4)  # TODO: find out what the 'multiple' is for

        f0, _, sp, ap = PyWorldPreprocessor.decompose_signal(padded, sampling_rate=Consts.sampling_rate)

        f0_converted = self._convert_pitch(f0, is_A=is_A)

        spectral_envelope = pw.code_spectral_envelope(sp, Consts.sampling_rate, Consts.number_of_mcpes)

        transposed = spectral_envelope.T

        normalized = self._normalize(transposed, is_A=is_A)

        return np.array([normalized]), (f0_converted, ap)

    def denormalize_and_save(self, signal, ap, f0, file_path: str, is_A: bool):
        squeezed = np.squeeze(signal)
        denormalized = self._denormalize(squeezed, is_A=is_A)
        transposed = denormalized.T
        transposed = np.ascontiguousarray(transposed)
        fft_len = pw.get_cheaptrick_fft_size(Consts.sampling_rate)
        decoded = pw.decode_spectral_envelope(transposed, Consts.sampling_rate, fft_len)
        synthesized = pw.synthesize(f0, decoded, ap, Consts.sampling_rate, Consts.frame_period_in_ms).astype(np.float32)

        sf.write(file=file_path, data=synthesized, samplerate=Consts.sampling_rate)
        # TODO: librosa do not support output anymore
        # librosa.output.write_wav(path=file_path,
        #                          y=synthesized,
        #                          sr=Consts.sampling_rate)

    @staticmethod
    def _unpack_normalizations(mcep, log_f0):
        mcep_mean = mcep['mean']
        mcep_std = mcep['std']

        log_f0_mean = log_f0['mean']
        log_f0_std = log_f0['std']

        return mcep_mean, mcep_std, log_f0_mean, log_f0_std

    @staticmethod
    def _pad_signal(signal, sr, frame_period, multiple):
        assert signal.ndim == 1  # TODO get rid of it
        number_of_frames = len(signal)

        parameter = (sr * frame_period / 1000)
        how_many_after_pad = int((np.ceil((np.floor(number_of_frames / parameter) + 1) / multiple + 1) * multiple - 1) * parameter)  # TODO: wtf
        difference = how_many_after_pad - number_of_frames
        pad_left = difference//2
        pad_right = difference - pad_left

        padded = np.pad(signal, (pad_left, pad_right), 'constant', constant_values=0)
        return padded

    def _convert_pitch(self, f0, is_A: bool):
        # mean_source, std_source, mean_target, std_target = 0
        log_f0 = np.log(f0)
        if is_A:
            mean_source = self.A_log_f0_mean
            std_source = self.A_log_f0_std
            mean_target = self.B_log_f0_mean
            std_target = self.B_log_f0_std
        else:
            mean_source = self.B_log_f0_mean
            std_source = self.B_log_f0_std
            mean_target = self.A_log_f0_mean
            std_target = self.A_log_f0_std

        # Logarithm Gaussian Normalization for Pitch Conversions
        f0_converted = np.exp((log_f0 - mean_source) / std_source * std_target + mean_target)

        return f0_converted

    def _normalize(self, spectral_envelope, is_A: bool):
        if is_A:
            mean = self.A_mcep_mean
            std = self.A_mcep_std
        else:
            mean = self.B_mcep_mean
            std = self.B_mcep_std
        normalized = (spectral_envelope - mean) / std

        return normalized

    def _denormalize(self, spectral_envelope, is_A: bool):
        if is_A:
            mean = self.A_mcep_mean
            std = self.A_mcep_std
        else:
            mean = self.B_mcep_mean
            std = self.B_mcep_std
        denormalized = (spectral_envelope * std) + mean

        return denormalized
