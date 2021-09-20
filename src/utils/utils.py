import torch


class Utils:
    @staticmethod
    def get_device():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Working on {device} device.")

        return device
