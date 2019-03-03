from namedtensor import ntorch
from namedtensor.nn import nn as nnn

class LSTMEncoder(nnn.Module):
    def __init__(self, TEXT,
                 embedding_dim=100,
                 hidden_dim=150,
                 num_layers=1,
                 dropout=0,
                 history=False,
                 bidirectional=False):
        super().__init__()
        self.history = history
        pad_idx = TEXT.vocab.stoi['<pad>']

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embedding_dim,
                                   padding_idx=pad_idx)

        self.lstm = nnn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=bidirectional) \
                        .spec("embedding", "srcSeqlen")

    def forward(self, batch_text):
        embedded = self.embed(batch_text)
        hidden_states, last_state = self.lstm(embedded)
        if self.history:
            return hidden_states
        else:
            return last_state
