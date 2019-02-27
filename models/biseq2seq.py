from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class BiSeq2Seq(nnn.Module):
    def __init__(self, encoder_fwd, encoder_back, decoder):
        super().__init__()
        self.encoder_fwd = encoder_fwd
        self.encoder_back = encoder_back
        self.decoder = decoder
    def forward(self, src, trg):
        hn_fwd, cn_fwd = self.encoder_fwd(src)
        hn_back, cn_back = self.encoder_back(src)
        hn = ntorch.cat([hn_fwd, hn_back], "embedding")
        cn = ntorch.cat([cn_fwd, cn_back], "embedding")
        return self.decoder((hn, cn), trg)


ce_loss = nnn.CrossEntropyLoss().spec("classes")

def seq2seq_loss_fn(model, batch):
    out = model(batch.src, batch.trg)
    return ce_loss(out, batch.trg)
