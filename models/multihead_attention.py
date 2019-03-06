import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn
from attention import Attention

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nnn.Module):
    """
    Takes query, key, value tuple and pass through n_head linear layers
    (with interim_dim_size output dimensions),
    attend, concatenate, and pass through one final linear layer.
    The layer maps input_dim_name of input_dim_size to
    input_dim_name with output_dim_size.
    """
    def __init__(self, n_heads, input_dim_name, input_dim_size,
                 interim_dim_size, output_dim_size, dropout=0, attn=None):
        super().__init__()
        self.linears = nnn.ModuleList([nnn.Linear(
            in_features=input_dim_size,
            out_features=interim_dim_size).spec(input_dim_name, input_dim_name)
        for _ in range(n_heads)])

        if attn is None:
            self.attn = Attention()
        else:
            self.attn = attn

        self.dropout = nnn.Dropout(dropout)
        self.linear_final = nnn.Linear(
            in_features=interim_dim_size * n_heads,
            out_features=output_dim_size).spec(input_dim_name, input_dim_name)

    def forward(self, query, key, embedding_dim, softmax_dim, mask=None, value=None):
        if value is None:
            value = key
        attended = [self.attn(l(query), l(key), embedding_dim,
            softmax_dim, mask=mask, value=l(value), dropout=self.dropout)
            for l in self.linears]
        a = [tup[0] for tup in attended]
        contexts = ntorch.cat([tup[1] for tup in attended], embedding_dim)
        return a, self.linear_final(contexts)


