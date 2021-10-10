import torch

from src.utils.utils import Utils

from sandpit.sandpit_utils import print_shape, print_value


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


if __name__ == '__main__':
    Utils.get_device()
    # axes_sandpit()
    # access_sandpit()
    # reshaping_sandpit()
    # element_wise_and_broadcasting_sandpit()
    # reduction_sandpit()

