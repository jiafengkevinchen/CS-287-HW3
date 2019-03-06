from namedtensor.nn import nn as nnn
from copy import deepcopy
from layernorm import LayerNorm
from namedtensor import ntorch, NamedTensor
from position_encoding import PositionalEncoding

class TransformerEncoder(nnn.Module):
    def __init__(self, embed_dim, position_dim, TEXT, layer, nlayers=3):
        super().__init__()
        size = layer.size
        pad_idx = TEXT.vocab.stoi['<pad>']
        assert embed_dim + position_dim == size, \
            "Embedding dimension + position_embedding must equal size"

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embed_dim,
                                   padding_idx=pad_idx)
        self.position_embed = PositionalEncoding(position_dim, "srcSeqlen")

        self.layers = nnn.ModuleList([deepcopy(layer) for _ in range(nlayers)])
        self.norm = LayerNorm(size, "embedding")
        self.embedding = nnn.embedding

    def forward(self, x):
        embed = self.embed(x)
        position_embed = self.position_embed(x)
        x = ntorch.cat([embed, position_embed], "embedding")
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


