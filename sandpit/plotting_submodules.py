import os
import torch

from src.model.submodules.glu import GLU
from src.utils.files_operator import FilesOperator
from consts import Consts


def _plot(tensor, model, file_name):
    path = os.path.join(Consts.submodules_graph_storage, file_name)
    print(f"{file_name}'s output shape: {tensor.shape}")
    FilesOperator.dump_torchviz_graph(tensor,
                                      dict(list(model.named_parameters())),
                                      path)


def plot_conv2d():
    x = torch.randn(158, 1, 24, 128)
    conv2d = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=[3, 4], stride=[1, 2])
    y = conv2d(x)

    _plot(y, conv2d, "conv2d")


def plot_instanceNorm2d():
    x = torch.randn(158, 128, 24, 1)
    instanceNorm2d = torch.nn.InstanceNorm2d(num_features=128, affine=True)
    y = instanceNorm2d(x)

    _plot(y, instanceNorm2d, "instanceNorm2d")


def plot_linear():
    x = torch.randn(1, 1, 1, 128)
    linear = torch.nn.Linear(in_features=128, out_features=1)
    y = linear(x)

    _plot(y, linear, "linear")


def plot_glu():
    x = torch.randn(158, 128, 24, 1)
    glu = GLU()
    y = glu(x)

    _plot(y, glu, "glu")


if __name__ == '__main__':
    # plot_conv2d()
    # plot_instanceNorm2d()
    plot_linear()
    plot_glu()
