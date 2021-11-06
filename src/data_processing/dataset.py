import numpy as np
import torch
from torch.utils.data import Dataset

from consts import Consts
from src.utils.files_operator import FilesOperator


class PreprocessedDataset(Dataset):
    def __init__(self, A_dataset_file, B_dataset_file, number_of_frames):
        self._A_file_content = FilesOperator.load_pickle_file(A_dataset_file)
        self._B_file_content = FilesOperator.load_pickle_file(B_dataset_file)

        self._number_of_frames = number_of_frames

        self._A_dataset, self._B_dataset = self._prepare_datasets()

    def __len__(self):
        return min(len(self._A_dataset), len(self._B_dataset))

    def __getitem__(self, idx):
        return self._A_dataset[idx], self._B_dataset[idx]

    def _prepare_datasets(self):
        number_of_samples = min(len(self._A_file_content), len(self._B_file_content))

        A_indexes = np.arange(len(self._A_file_content))
        B_indexes = np.arange(len(self._B_file_content))

        A_indexes_shuffled = np.random.permutation(A_indexes)
        B_indexes_shuffled = np.random.permutation(B_indexes)

        A_indexes_shrinked = A_indexes_shuffled[:number_of_samples]
        B_indexes_shrinked = B_indexes_shuffled[:number_of_samples]

        A_ready = list()
        B_ready = list()
        for (A_idx, B_idx) in zip(A_indexes_shrinked, B_indexes_shrinked):
            A_data = self._A_file_content[A_idx]
            B_data = self._B_file_content[B_idx]

            A_data_shrinked = self._get_random_frames_chunk(A_data)
            A_ready.append(A_data_shrinked)

            B_data_shrinked = self._get_random_frames_chunk(B_data)
            B_ready.append(B_data_shrinked)

        A_numpy = np.array(A_ready)
        B_numpy = np.array(B_ready)

        A_torch = torch.from_numpy(A_numpy)
        B_torch = torch.from_numpy(B_numpy)

        # '.float()' because network for some reason is expecting float32 instead of float64
        # TODO
        return A_torch.float(), B_torch.float()

    def _get_random_frames_chunk(self, data):  # TODO: maybe get rid of this
        all_data_frames = data.shape[1]
        start = np.random.randint(all_data_frames - self._number_of_frames + 1)
        end = start + self._number_of_frames

        return data[:, start:end]


if __name__ == '__main__':
    dataset = PreprocessedDataset(A_dataset_file=Consts.A_preprocessed_dataset_file_path,
                                  B_dataset_file=Consts.B_preprocessed_dataset_file_path,
                                  number_of_frames=Consts.number_of_frames)

    print(len(dataset[0]))
    print(len(dataset[0][0]))
    print(dataset[0][0].shape)
    print(len(dataset[0][0][0]))
