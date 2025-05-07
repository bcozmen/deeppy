import torch
import torch.nn as nn

from deeppy.utils import print_args


class SqueezeLastDimention(nn.Module):
    print_args = classmethod(print_args)
    dependencies = []
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[:-2] + (-1,))