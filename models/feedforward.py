from namedtensor.nn import nn as nnn

class FeedForward(nnn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dim_name, dropout=0.1):
        super().__init__()
        self.w_1 = nnn.Linear(in_features=d_model,
            out_features=d_ff).spec(dim_name, dim_name)
        self.w_2 = nnn.Linear(in_features=d_ff,
            out_features=d_model).spec(dim_name, dim_name)
        self.dropout = nnn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
