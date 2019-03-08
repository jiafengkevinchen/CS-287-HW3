import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn
from attention import Attention
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTMDecoder(nnn.Module):
    def __init__(self, TEXT,
                 embedding_dim=100,
                 hidden_dim=150,
                 num_layers=1,
                 dropout=0):
        super().__init__()

        pad_idx = TEXT.vocab.stoi['<pad>']

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embedding_dim,
                                   padding_idx=pad_idx)

        self.lstm = nnn.LSTM(input_size=embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout) \
            .spec("embedding", "trgSeqlen")

        self.w = nnn.Linear(in_features=hidden_dim,
                            out_features=len(TEXT.vocab)) \
            .spec("embedding", "classes")

    def forward(self, init_state, batch_text):
        embedded = self.embed(batch_text)
        hidden_states, (hn, cn) = self.lstm(embedded, init_state)
        log_probs = self.w(hidden_states)
        return log_probs


class LSTMDecoderAttn(nnn.Module):
    def __init__(self, TEXT,
                 embedding_dim=100,
                 hidden_dim=150,
                 num_layers=1,
                 dropout=0,
                 attn_normalize=True):
        super().__init__()

        pad_idx = TEXT.vocab.stoi['<pad>']

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embedding_dim,
                                   padding_idx=pad_idx)

        self.lstm = nnn.LSTM(input_size=embedding_dim + hidden_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout) \
            .spec("embedding", "trgSeqlen")

        self.w = nnn.Linear(in_features=hidden_dim,
                            out_features=len(TEXT.vocab)) \
            .spec("embedding", "classes")

        self.attention = []

        self.attn = Attention(attn_normalize)

    def forward(self, init_state, batch_text):
        H,  (ht, ct) = init_state
        ht, ct = (unsqueeze(flatten(ht, "layers", "embedding"), "layers"),
                  unsqueeze(flatten(ct, "layers", "embedding"), "layers"))

        embedded = self.embed(batch_text)

        hidden_states = []
        attention_weights = []
        ht_flat = flatten(ht, "layers", "embedding")

        for t in range(embedded.shape["trgSeqlen"]):

            a, context = self.attn(ht_flat, H, "embedding", "srcSeqlen")
            context = unsqueeze(context, "trgSeqlen")
            word_t = embedded[{'trgSeqlen': slice(t, t+1)}]
            lstm_input = ntorch.cat([word_t, context], "embedding")
            output, (ht, ct) = self.lstm(lstm_input, (ht, ct))
            if not self.training:
                attention_weights.append(a)
            ht_flat = flatten(ht, "layers", "embedding")
            hidden_states.append(ht_flat)

        if not self.training:
            self.attention = ntorch.stack(attention_weights, 'trgSeqlen')
        return self.w(ntorch.stack(hidden_states, "trgSeqlen"))
