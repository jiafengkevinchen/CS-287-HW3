import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn
from attention import Attention
from layernorm import LayerNorm

class SublayerConnection(nnn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, dim):
        super().__init__()
        self.norm = LayerNorm(size, dim)
        self.dropout = nnn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        out = sublayer(self.norm(x))
        if out is tuple:
            a, context = out
            return a, x + self.dropout(context)
        else:
            return x + self.dropout(out)
