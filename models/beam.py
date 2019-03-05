from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class Beam(nnn.Module):
    def __init__(self, beam_size=3):
        super().__init__()

        self.prev = []
        self.scores = []
        self.pred = []

    def forward(self, model, src):
        encoded = model.encoder(src)
        