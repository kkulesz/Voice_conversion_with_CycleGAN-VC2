import torch
import torch.nn as nn

from src.utils.utils import Utils
from src.utils.files_operator import FilesOperator

from sandpit.sandpit_utils import print_tensor, print_grad


def autograd():
    loss_fn = nn.MSELoss()

    x1 = torch.tensor([.1, .1, .1])
    w1 = torch.tensor([.5, .5, .5], requires_grad=True)
    w2 = torch.tensor([.2, .3, .4], requires_grad=True)
    optimizer = torch.optim.Adam([w1, w2], lr=.01)
    y = torch.tensor([1.0, 1.0, 1.0])

    print_tensor("w1 before iteration:", w1)
    print_grad("w1.grad init", w1)

    y_pred = x1 + w1 + w2

    loss = loss_fn(y_pred, y)
    print_grad("w1.grad after loss_fn", w1)

    loss.backward()
    print_grad("w1.grad after backward", w1)
    FilesOperator.dump_torchviz_graph(loss, {"w1": w1, "w2": w2, "loss": loss})

    optimizer.step()
    print_grad("w1.grad after step", w1)
    print_tensor("w1 after iteration", w1)

    optimizer.zero_grad()
    print_grad("w1.grad after zero_grad", w1)


def detach():
    x1 = torch.tensor([.1])
    w1 = torch.tensor([.8], requires_grad=True)
    y = torch.tensor([1.0])

    y_pred = x1 + w1

    loss = torch.abs(y_pred - y)
    FilesOperator.dump_torchviz_graph(loss, {"w1": w1, "loss": loss})
    loss_detached = loss.detach()
    FilesOperator.dump_torchviz_graph(loss_detached, {"w1": w1, "loss_detached": loss_detached}, "GRAPH_DETACHED")


def no_grad():
    loss_fn = nn.MSELoss()

    x1 = torch.tensor([.1, .1, .1])
    w1 = torch.tensor([.5, .5, .5], requires_grad=True)
    w2 = torch.tensor([.2, .3, .4], requires_grad=True)
    optimizer = torch.optim.Adam([w1, w2], lr=.01)
    y = torch.tensor([1.0, 1.0, 1.0])

    with torch.no_grad():
        y_pred = x1 + w1 + w2
        loss = loss_fn(y_pred, y)
        # loss.backward() #cannot use because tensor does not have grad_fn
        FilesOperator.dump_torchviz_graph(loss, {"w1": w1, "w2": w2, "loss": loss})
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    Utils.get_device()

    # autograd()
    # detach()
    no_grad()
