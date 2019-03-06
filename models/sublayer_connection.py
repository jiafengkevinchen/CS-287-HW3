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
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nnn.Dropout(dropout)

    def forward(self, x, sublayer, dim):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x, dim)))
