import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Attention(nnn.Module):
    def forward(self, h, H, embedding_dim, src_seqlen_dim):
        log_weights = (h.dot(embedding_dim, H) / (H.size(embedding_dim) ** .5))
        a = log_weights.softmax(src_seqlen_dim)

        return (a, (a * H).sum(src_seqlen_dim))
