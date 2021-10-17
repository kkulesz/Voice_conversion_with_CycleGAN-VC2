import torch
import os

_sandpit_path = os.path.dirname(os.path.abspath(__file__))

sandpit_datasets_dir_path = os.path.join(_sandpit_path, "data")
sandpit_output_dir_path = os.path.join(_sandpit_path, "output")


def print_tensor(message: str, t: torch.Tensor):
    print("{0: <30}: {1}".format(message, t))


def print_shape(message: str, t: torch.Tensor):
    print("{0: <20}: {1}".format(message, t.shape))


def print_value(message: str, val: any):
    print("{0: <20}: {1}".format(message, val))


def print_grad(message: str, t: torch.Tensor):
    print("{0: <25}: {1}".format(message, t.grad))
