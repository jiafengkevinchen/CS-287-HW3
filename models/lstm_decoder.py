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
                 dropout=0):
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

        self.attn = Attention()

    def forward(self, init_state, batch_text):
        H,  (ht, ct) = init_state

        embedded = self.embed(batch_text)

        hidden_states = []
        attention_weights = []
        ht_flat = flatten(ht, "layers", "embedding")

        for t in range(embedded.shape["trgSeqlen"]):

            a, context = self.attn(ht_flat, H, "embedding", "srcSeqlen")
            context = unsqueeze(context, "trgSeqlen")
            word_t = embedded[{'trgSeqlen': slice(t, t+1)}]
            lstm_input = ntorch.cat([word_t, context], "embedding")
            output, (ht, ct) = self.lstm(lstm_input, (unsqueeze(ht_flat, "layers"), ct))
            if not self.training:
                attention_weights.append(a)
            ht_flat = flatten(ht, "layers", "embedding")
            hidden_states.append(ht_flat)

        if not self.training:
            self.attention = ntorch.stack(attention_weights, 'trgSeqlen')
        return self.w(ntorch.stack(hidden_states, "trgSeqlen"))

        # first_input = ntorch.cat([embedded[{"trgSeqlen": slice(0, 1)}],
        #                           ntorch.zeros((1, init_state.shape["batch"],
        #                                         init_state.shape["embedding"]),
        #                                        names=("trgSeqlen", "batch", "embedding")).to(device)],
        #                          "embedding")
        # output, (hn, cn) = self.lstm(first_input, )

        # (hn, cn)]
        # for i in range(2, embedded.shape["trgSeqlen"] + 1):
        #     last_hidden = hidden_states[-1]
        #     attention = init_state.dot("embedding", last_hidden[0].squeeze("layers")) \
        #                           .softmax("srcSeqlen")
        #     if self.training is False:
        #         attention_weights.append(attention)
        #     context = attention.dot("srcSeqlen", init_state)
        #     context = NamedTensor(context.values.unsqueeze(-1),
        #                           names=(*context.shape.keys(), 'trgSeqlen'))
        #     prev_word = embedded[{"trgSeqlen": slice(i - 1, i)}]
        #     lstm_input = ntorch.cat([prev_word, context], "embedding")
        #     output, (hn, cn) = self.lstm(lstm_input, last_hidden)
        #     hidden_states.append((hn, cn))

        # # save attention weights when evaluating
        # if self.training is False:
        #     self.attention = ntorch.stack(attention_weights, 'trgSeqlen')

        # return self.w(ntorch.cat([hn for (hn, cn)
        #                           in hidden_states], "layers")
        #               .rename("layers", "trgSeqlen"))
