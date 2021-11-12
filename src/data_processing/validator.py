import librosa
import pyworld as pw
import numpy as np
import soundfile as sf

from consts import Consts
from src.utils.files_operator import FilesOperator
from src.data_processing.processing_utils import ProcessingUtils


class Validator:
    def __init__(self, A_cache_dir, B_cache_dir):
        (A_mcep, A_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(A_cache_dir)
        (B_mcep, B_log_f0) = FilesOperator.load_preprocessed_data_normalization_files(B_cache_dir)

        self.A_mcep_mean, self.A_mcep_std, self.A_log_f0_mean, self.A_log_f0_std = \
            Validator._unpack_normalizations(A_mcep, A_log_f0)
        self.B_mcep_mean, self.B_mcep_std, self.B_log_f0_mean, self.B_log_f0_std = \
            Validator._unpack_normalizations(B_mcep, B_log_f0)

        print(f"- A_ds: logf0_Mean: {self.A_log_f0_mean: .4f}, logf0_Std: {self.A_log_f0_std: .4f}")
        # print(f"A_ds: MCEP_Mean: {self.A_mcep_mean}, MCEP_Std: {self.A_mcep_std}")

        print(f"- B_ds: logf0_Mean: {self.B_log_f0_mean: .4f}, logf0_Std: {self.B_log_f0_std: .4f}")
        # print(f"B_ds: MCEP_Mean: {self.B_mcep_mean}, MCEP_Std: {self.B_mcep_std}")

    def load_and_normalize(self, file_path: str, is_A: bool):
        signal, _ = librosa.load(file_path, sr=Consts.sampling_rate, mono=True)

        padded = ProcessingUtils.pad_signal(signal,
                                            sr=Consts.sampling_rate,
                                            frame_period=Consts.frame_period_in_ms,
                                            multiple=4)  # TODO: find out what the 'multiple' is for
        f0, _, sp, ap = ProcessingUtils.decompose_signal(padded, sampling_rate=Consts.sampling_rate)
        f0_converted = self._convert_pitch(f0, is_A=is_A)
        mcep = pw.code_spectral_envelope(sp, Consts.sampling_rate, Consts.number_of_mcpes_cycle_gan_2)
        transposed = mcep.T
        normalized = self._normalize(transposed, is_A=is_A)

        return np.array([normalized]), (f0_converted, ap)

    def denormalize_and_save(self, signal, ap, f0, file_path: str, is_A: bool):
        sr = Consts.sampling_rate

        squeezed = np.squeeze(signal)
        denormalized = self._denormalize(squeezed, is_A=is_A)
        transposed = denormalized.T
        contiguous_transposed = np.ascontiguousarray(transposed)

        fft_len = pw.get_cheaptrick_fft_size(sr)
        decoded = pw.decode_spectral_envelope(contiguous_transposed, sr, fft_len)
        synthesized = pw.synthesize(f0, decoded, ap, sr, Consts.frame_period_in_ms).astype(np.float32)

        sf.write(file=file_path, data=synthesized, samplerate=sr)  # librosa does not handle output anymore

    @staticmethod
    def _unpack_normalizations(mcep, log_f0):
        mcep_mean = mcep['mean']
        mcep_std = mcep['std']

        log_f0_mean = log_f0['mean']
        log_f0_std = log_f0['std']

        return mcep_mean, mcep_std, log_f0_mean, log_f0_std

    def _convert_pitch(self, f0, is_A: bool):
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

        return ProcessingUtils.gaussian_normalization(log_f0, mean_source, std_source, mean_target, std_target)

    def _normalize(self, mcep, is_A: bool):
        mean, std = self._get_mean_and_std(is_A)

        return ProcessingUtils.normalize_mcep(mcep, mean, std)

    def _denormalize(self, mcep, is_A: bool):
        mean, std = self._get_mean_and_std(is_A)
        return ProcessingUtils.denormalize_mcep(mcep, mean, std)

    def _get_mean_and_std(self, is_A):
        if is_A:
            return self.A_mcep_mean, self.A_mcep_std
        else:
            return self.B_mcep_mean, self.B_mcep_std


if __name__ == '__main__':
    validator = Validator(A_cache_dir=Consts.A_cache_directory_path,
                          B_cache_dir=Consts.B_cache_directory_path)

    input_file = './input.wav'
    output_file = './output.wav'
    loaded_signal, (f0, ap) = validator.load_and_normalize(file_path=input_file, is_A=True)
    validator.denormalize_and_save(signal=loaded_signal,
                                   f0=f0,
                                   ap=ap,
                                   file_path=output_file,
                                   is_A=True)
    # result: .wav files are a little different, but the same happens in follow-up works, so it should not be the problem
