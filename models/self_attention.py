import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn
from attention import Attention

class SelfAttention(Attention):
    def __init__(self, *args):
        super().__init__(*args)
    def forward(self, query, key, embedding_dim, softmax_dim, mask=None,
        value=None, **kwargs):
        # Source is key,values, Target is query
        target_name = f"trg{softmax_dim}"
        trg = key.rename(softmax_dim, target_name)
        if mask is not None:
            mask = mask.rename("src", softmax_dim).rename("trg", target_name)

        a, context = super().forward(query, trg, embedding_dim, softmax_dim,
            mask, value=trg, **kwargs)

        return a, context.rename(target_name, softmax_dim)


