import torch
import torch.nn as nn

class SqueezeLastDimention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[:-2] + (-1,))