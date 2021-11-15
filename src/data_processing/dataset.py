import torch
import numpy as np
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, A_dataset, B_dataset, number_of_frames):
        self.A_dataset = A_dataset
        self.B_dataset = B_dataset
        self.length = min(len(self.A_dataset), len(self.B_dataset))
        self.number_of_frames = number_of_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        A_idx = np.random.randint(len(self.A_dataset))
        B_idx = np.random.randint(len(self.B_dataset))

        if self.number_of_frames > 0:
            A_sample = self._get_random_frames_chunk(self.A_dataset[A_idx])
            B_sample = self._get_random_frames_chunk(self.B_dataset[B_idx])
        else:
            A_sample = self.A_dataset[A_idx]
            B_sample = self.B_dataset[B_idx]

        A_ready = torch.Tensor(A_sample).float()
        B_ready = torch.Tensor(B_sample).float()
        return A_ready, B_ready

    def _get_random_frames_chunk(self, data):
        all_data_frames = data.shape[1]
        start = np.random.randint(all_data_frames - self.number_of_frames + 1)
        end = start + self.number_of_frames

        return data[:, start:end]
