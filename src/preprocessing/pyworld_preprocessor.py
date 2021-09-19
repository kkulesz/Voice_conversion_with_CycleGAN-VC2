import pyworld as pw

from src.preprocessing.preprocessor import Preprocessor


class PyWorldPreprocessor(Preprocessor):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms, number_of_frames):
        super(PyWorldPreprocessor, self).__init__(number_of_mceps, sampling_rate, frame_period_in_ms, number_of_frames)

    def preprocess(self, data_directory):
        """
        todo
        """
        signals = self._load_signals(data_directory)

        return signals

    def encode_single_speaker_signals(self, speaker_signals):
        """
        todo
        """
        pass


if __name__ == '__main__':
    preprocessor = PyWorldPreprocessor(1, 1, 1, 1)
