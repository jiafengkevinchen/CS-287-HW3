from namedtensor.nn import nn as nnn
from copy import deepcopy
from layernorm import LayerNorm
from namedtensor import ntorch, NamedTensor
from position_encoding import PositionalEncoding


class TransformerDecoder(nnn.Module):
    def __init__(self, embed_dim, position_dim, TEXT, layer, nlayers=3):
        super().__init__()
        size = layer.size
        pad_idx = TEXT.vocab.stoi['<pad>']
        assert embed_dim + position_dim == size, \
            "Embedding dimension + position_embedding must equal size"

        self.embed = nnn.Embedding(num_embeddings=len(TEXT.vocab),
                                   embedding_dim=embed_dim,
                                   padding_idx=pad_idx)
        self.position_embed = PositionalEncoding(position_dim, "trgSeqlen")

        self.layers = nnn.ModuleList([deepcopy(layer) for _ in range(nlayers)])
        self.norm = LayerNorm(size, "embedding")
        self.attn_weights = []
        self.w = nnn.Linear(in_features=size, out_features=len(TEXT.vocab)).spec("embedding", "classes")

    def forward(self, encoded, trg):
        embed = self.embed(trg)
        position_embed = self.position_embed(trg)
        x = ntorch.cat([embed, position_embed], "embedding")
        for layer in self.layers:
            a, x = layer(encoded, x)
            if not self.training:
                self.attn_weights.append(a)
        if not self.training:
            self.attn_weights = ntorch.stack(self.attn_weights, "layers")
        x = self.norm(x)
        return self.w(x)g




