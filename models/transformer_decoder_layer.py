from namedtensor.nn import nn as nnn
from sublayer_connection import SublayerConnection
from namedtensor import ntorch, NamedTensor


class TransformerDecoderLayer(nnn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, size, dropout=.3):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = nnn.ModuleList([
                SublayerConnection(size, dropout, "embedding")
                for _ in range(3)])
        self.size = size


    def forward(self, encoded, trg):
        trglen = trg.size("trgSeqlen")
        mask = ntorch.triu(ntorch.ones(trglen, trglen, names=('src','trg'))
                           .to(trg.values.device),
                           dims=("src","trg"))
        a, x = self.sublayer[0](trg, lambda x: self.self_attn(x, x, "embedding", "trgSeqlen", mask=mask))
        a, x = self.sublayer[1](trg, lambda x: self.src_attn(x, encoded, "embedding", "srcSeqlen"))

        return a, self.sublayer[2](x, self.feed_forward)

