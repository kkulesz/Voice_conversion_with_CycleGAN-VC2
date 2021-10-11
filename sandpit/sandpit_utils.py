import torch
import os
import torchviz

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


def print_tensor(message: str, t: torch.Tensor):
    print("{0: <30}: {1}".format(message, t))


def print_shape(message: str, t: torch.Tensor):
    print("{0: <20}: {1}".format(message, t.shape))


def print_value(message: str, val: any):
    print("{0: <20}: {1}".format(message, val))


def print_grad(message: str, t: torch.Tensor):
    print("{0: <25}: {1}".format(message, t.grad))


def dump_torchviz_graph(what_to_graph, params={}, file_name="GRAPH"):
    torchviz.make_dot(what_to_graph, params=params).render(file_name, format="png")
    os.remove(file_name)
