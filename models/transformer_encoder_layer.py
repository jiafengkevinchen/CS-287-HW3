from namedtensor.nn import nn as nnn
from sublayer_connection import SublayerConnection

class TransformerEncoderLayer(nnn.Module):
    def __init__(self, attn, feed_forward, n_heads=5, size=100, dropout=.3):
        super().__init__()
        self.attn = attn
        self.feed_forward = feed_forward

        self.sublayer = nnn.ModuleList([
                SublayerConnection(size, dropout)
            ] for _ in range(2))
        self.size = size


    def forward(self, src):
        x = self.sublayer[0](src,
            lambda src: self.attn(src, src, "embedding", "srcSeqlen"))
        return self.sublayer[1](x, self.feed_forward)

