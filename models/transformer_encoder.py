from namedtensor.nn import nn as nnn
from copy import deepcopy
from layernorm import LayerNorm
from namedtensor import ntorch, NamedTensor

MAX_LEN = 20


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

        self.position_embed = nnn.Embedding(num_embeddings=MAX_LEN + 1,
                                   embedding_dim=position_dim,
                                   padding_idx=MAX_LEN)

        self.layers = nnn.ModuleList([deepcopy(layer) for _ in range(nlayers)])
        self.norm = LayerNorm(size, "embedding")

    def forward(self, x):
        pos = ntorch.ones(*x.shape.values(), names=[*x.shape.keys()]).to(x.values.device)
        pos_vec = ntorch.arange(x.size("srcSeqlen"), names="srcSeqlen")
        pos_vec[pos_vec > MAX_LEN] = MAX_LEN
        embed = self.embed(x)
        position_embed = self.position_embed((pos * pos_vec.to(x.values.device)).long())


        x = ntorch.cat([embed, position_embed], "embedding")
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


