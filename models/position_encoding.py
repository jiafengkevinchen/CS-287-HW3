import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn


class PositionalEncoding(nnn.Module):
    def __init__(self, size, seqlen_name):
        super().__init__()
        self.position_embedding = ntorch.zeros(size, MAX_LEN, names=("embedding", seqlen_name))
        self.register_parameter("positionembed", self.position_embedding)
        self.seqlen_name = seqlen_name
        self.size = size
    def forward(self, x):
        length = min(x.size(self.seqlen_name), MAX_LEN)
        mask = ntorch.zeros(*x.shape.values(), self.size ,
            names=[*x.shape.keys(), 'embedding'],
            device=self.position_embedding.values.device)
        mask[{self.seqlen_name: slice(0, length)}] = self.position_embedding[{self.seqlen_name: slice(0, length)}]
        return mask.to(self.position_embedding.values.device)
