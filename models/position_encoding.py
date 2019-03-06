import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

MAX_LEN = 30

class PositionalEncoding(nnn.Module):
    def __init__(self, size, seqlen_name):
        super().__init__()
        self.position_embedding = ntorch.zeros(size, MAX_LEN, names=("embedding", seqlen_name))
        self.register_parameter("positionembed", self.position_embedding)
        self.seqlen_name = seqlen_name
    def forward(self, x):
        length = x.size(self.seqlen_name)
        mask = ntorch.zeros(*x.shape.values(), names=[*x.shape.keys()])
        mask[{self.seqlen_name: slice(0, length)}] = self.position_embedding
        return mask
