import torch
import torch.nn as nn

class SqueezeLastDimention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[:-2] + (-1,))
class SqueezeLastDimention2Inputs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z1,z2):
        z1 = z1.view(z1.shape[:-2] + (-1,))
        z2 = z2.view(z2.shape[:-2] + (-1,))
        return torch.cat([z1, z2], dim=-1)