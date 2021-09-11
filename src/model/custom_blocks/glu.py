import torch


class GLU(torch.nn.Module):
    """
    TODO
    """
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
