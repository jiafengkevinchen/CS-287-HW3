from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class Seq2Seq(nnn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src):
        encoded = self.encoder(src)
        return self.decoder(encoded)
