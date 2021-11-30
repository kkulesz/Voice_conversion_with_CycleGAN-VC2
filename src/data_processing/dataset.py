import os
import torch
import numpy as np
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    def __init__(self, A_dataset, B_dataset, number_of_frames):
        self.A_dataset = A_dataset
        self.B_dataset = B_dataset
        self.length = min(len(self.A_dataset), len(self.B_dataset))
        self.number_of_frames = number_of_frames

        print(f"SIZE OF DATASET = {self.length}")

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


class LightningValidationDataset(Dataset):
    """
    dummy dataset for PytorchLighting
    """
    def __init__(self, A_dir, B_dir, validator):
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.A_files = os.listdir(A_dir)
        self.B_files = os.listdir(B_dir)

        self.validator = validator

        self.length = min(len(self.A_files), len(self.B_files))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        A_file_name = self.A_files[idx]
        B_file_name = self.B_files[idx]
        A_file = os.path.join(self.A_dir, A_file_name)
        B_file = os.path.join(self.B_dir, B_file_name)

        A = self.validator.load_and_normalize(A_file, is_A=True)
        B = self.validator.load_and_normalize(B_file, is_A=False)
        return (A, A_file_name), (B, B_file_name)

