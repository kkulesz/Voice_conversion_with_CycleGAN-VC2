import os
import librosa
from abc import ABC
from abc import abstractmethod


class Preprocessor(ABC):
    def __init__(self, number_of_mceps, sampling_rate, frame_period_in_ms):
        self._number_of_mceps = number_of_mceps
        self._sampling_rate = sampling_rate
        self._frame_period_in_ms = frame_period_in_ms

    @abstractmethod
    def preprocess(self, data_directory: str, cache_directory: str) -> None:
        pass

    def _load_signals(self, data_directory: str):
        signals = list()
        for file in os.listdir(data_directory):
            file_path = os.path.join(data_directory, file)
            signal, _ = librosa.load(file_path, sr=self._sampling_rate, mono=True)
            signals.append(signal)

        return signals
