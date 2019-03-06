from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class Seq2Seq(nnn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg):
        encoded = self.encoder(src)
        return self.decoder(encoded, trg)


ce_loss = nnn.CrossEntropyLoss().spec("classes")

# def seq2seq_loss_fn(model, batch):
#     out = model(batch.src, batch.trg)
#     length = batch.trg.size('trgSeqlen')
#     return ce_loss(out[{'trgSeqlen': slice(0, length-1)}],
#                    batch.trg[{'trgSeqlen': slice(1, length)}])


def seq2seq_loss_fn(model, batch):
    length = batch.trg.size('trgSeqlen')
    out = model(batch.src, batch.trg)[{'trgSeqlen': slice(0, length - 1)}]
    trg = batch.trg[{'trgSeqlen': slice(1, length)}]
    mask = (trg != pad_idx_EN).float()
    
    n = trg.values.numel()
    loss = ce_loss(mask * out, trg)
    
    return (loss * n - np.log(V_EN) * (n - mask.sum().item())) / mask.sum().item()
