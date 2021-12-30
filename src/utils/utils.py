import torch
import numpy as np
import os
import random


class Utils:
    @staticmethod
    def get_device():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Working on {device} device.")

        return device

    @staticmethod
    def seed_torch(seed=2137):
        #  taken from: https://github.com/pytorch/pytorch/issues/7068
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
