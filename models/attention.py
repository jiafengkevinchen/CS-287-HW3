import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Attention(nnn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, embedding_dim, softmax_dim,
                mask=None, value=None, dropout=None,
                self_attention=False):
        log_weights = (query.dot(embedding_dim, key) / (key.size(embedding_dim) ** .5))

        if mask is not None:
            log_weights = log_weights.masked_fill_(mask == 0, -1e9)

        if value is None:
            value = key

        a = log_weights.softmax(softmax_dim)

        if dropout is not None:
            a = dropout(a)

        return (a, (a * value).sum(softmax_dim))
