from namedtensor.nn import nn as nnn
from copy import deepcopy
from layernorm import LayerNorm

class TransformerEncoder(nnn.Module):
    def __init__(self, layer, nlayers=3):
        super().__init__()
        self.layers = nnn.ModuleList([deepcopy(layer) for _ in range(nlayers)])
        self.norm = LayerNorm(layer.size, "embedding")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


