from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class BiEncoder(nnn.Module):
    def __init__(self, encoder_fwd, encoder_back):
        super().__init__()
        self.encoder_fwd = encoder_fwd
        self.encoder_back = encoder_back

    def forward(self, src):
        hn_fwd, cn_fwd = self.encoder_fwd(src)
        hn_back, cn_back = self.encoder_back(
            src[{'seqlen':slice(None, None, -1)}])
        hn = ntorch.cat([hn_fwd, hn_back], "embedding")
        cn = ntorch.cat([cn_fwd, cn_back], "embedding")
        return (hn, cn)
