import torch
import torch.nn as nn

from deeppy.utils import print_args


class MaskedTransformerEncoder(nn.Module):
    print_args = classmethod(print_args)
    dependencies = []
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.TransformerEncoder(**kwargs)
    def forward(self, x):
        sz = x.shape[1]
        mask = torch.log(torch.tril(torch.ones(sz,sz))).to(x.device)
        return self.encoder(x,mask = mask)