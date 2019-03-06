import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

class LayerNorm(nnn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nnn.Parameter(torch.ones(features))
        self.b_2 = nnn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x, dim):
        mean = x.mean(dim)
        std = x.std(dim)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
