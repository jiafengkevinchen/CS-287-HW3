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

def seq2seq_loss_fn(model, batch):
    out = model(batch.src, batch.trg)
    return ce_loss(out, batch.trg)
