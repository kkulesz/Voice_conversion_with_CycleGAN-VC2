import torch
import torch.nn as nn


def print_shape(message: str, t: torch.Tensor):
    print("{0: <20}: {1}".format(message, t.shape))


def print_value(message: str, val: any):
    print("{0: <20}: {1}".format(message, val))


def axes_sandpit():
    t = torch.ones(3, 3, 3)
    print(t)
    print_shape("Axes", t)


def access_sandpit():
    t = torch.randn(3, 3, 3)
    print("First element: {0}".format(t[0][0][0]))
    print("Number of element with .numel(): {0}".format(t.numel()))


def reshaping_sandpit():
    array = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    base = torch.Tensor(array)
    print_shape("Base tensor", base)

    reshaped = base.reshape(1, 9)
    print_shape("After reshape (1, 9)", reshaped)

    unsqueezed = reshaped.squeeze()
    print_shape("Squeezed", unsqueezed)

    squeezed = unsqueezed.unsqueeze(1)
    print_shape("Unqueezed", squeezed)

    print("Reshaping back to (3,3)...")
    reshaped = squeezed.reshape(3, 3)

    flattened = reshaped.flatten()
    print_shape("Flattened", flattened)


def element_wise_and_broadcasting_sandpit():
    twos = torch.ones(3, 3, 3) * 2  # broadcasting
    threes = torch.ones(3, 3) * 3  # broadcasting

    print(twos * threes)  # broadcasting and then element-wise multiplying


def reduction_sandpit():
    base = torch.randn(3, 3, 3)
    # print(base)
    print(base.flatten())
    print_value(".sum()", base.sum())
    print_value(".prod()", base.prod())
    print_value(".mean()", base.mean())
    print_value(".std()", base.std())
    print_value(".max()", base.max())
    print_value(".argmax()", base.argmax())
    print_value(".max().item()", base.max().item())


def checking_if_switching_device_copies_tensor():
    cuda = torch.zeros(1).cuda()
    print_value("Initial cuda:", cuda.sum())
    cpu = cuda.cpu()
    print_value("Cpu:", cpu.sum())
    print_value("Initial cuda:", cuda.sum())  # not changes


if __name__ == '__main__':
    # axes_sandpit()
    # access_sandpit()
    # reshaping_sandpit()
    # element_wise_and_broadcasting_sandpit()
    # reduction_sandpit()
    checking_if_switching_device_copies_tensor()
