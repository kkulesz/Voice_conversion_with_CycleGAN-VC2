import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from src.consts import Consts


class PreprocessedDataset(Dataset):
    def __init__(self, A_dataset_file, B_dataset_file, number_of_frames):
        with open(A_dataset_file, 'rb') as A_file:
            self._A_dataset = pickle.load(A_file)
        with open(B_dataset_file, 'rb') as B_file:
            self._B_dataset = pickle.load(B_file)
        self._number_of_frames = number_of_frames

        self._A_dataset, self._B_dataset = self._increase_randomness()

    def __len__(self):
        return min(len(self._A_dataset), len(self._B_dataset))

    def __getitem__(self, idx):
        return self._A_dataset[idx], self._B_dataset[idx]

    def _increase_randomness(self):
        number_of_samples = min(len(self._A_dataset), len(self._B_dataset))  # __len__()?

        A_indexes = np.arange(len(self._A_dataset))
        B_indexes = np.arange(len(self._B_dataset))

        A_indexes_shuffled = np.random.permutation(A_indexes)
        B_indexes_shuffled = np.random.permutation(B_indexes)

        A_indexes_shrinked = A_indexes_shuffled[:number_of_samples]
        B_indexes_shrinked = B_indexes_shuffled[:number_of_samples]

        A_ready = list()
        B_ready = list()
        for (A_idx, B_idx) in zip(A_indexes_shrinked, B_indexes_shrinked):
            A_data = self._A_dataset[A_idx]
            B_data = self._B_dataset[B_idx]

            A_data_shrinked = self._get_random_frames_chunk(A_data)
            A_ready.append(A_data_shrinked)

            B_data_shrinked = self._get_random_frames_chunk(B_data)
            B_ready.append(B_data_shrinked)

        return np.array(A_ready), np.array(B_ready)

    def _get_random_frames_chunk(self, data):
        all_data_frames = data.shape[1]
        start = np.random.randint(all_data_frames - self._number_of_frames + 1)
        end = start + self._number_of_frames

        return data[:, start:end]


if __name__ == '__main__':
    dataset = PreprocessedDataset(A_dataset_file=Consts.A_preprocessed_dataset_file,
                                  B_dataset_file=Consts.B_preprocessed_dataset_file,
                                  number_of_frames=Consts.number_of_frames)

    print(len(dataset[0]))
    print(len(dataset[0][0]))
    print(dataset[0][0].shape)
    print(len(dataset[0][0][0]))


