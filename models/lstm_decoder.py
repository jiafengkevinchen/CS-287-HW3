from torch import nn
from namedtensor import ntorch
from namedtensor.nn import nn as nnn

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

        self.lstm = nnn.LSTM(input_size=embedding_dim + 2 * hidden_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout) \
                        .spec("embedding", "trgSeqlen")

        self.w = nnn.Linear(in_features=hidden_dim,
                            out_features=len(TEXT.vocab)) \
                        .spec("embedding", "classes")


    def forward(self, init_state, batch_text):
        embedded = self.embed(batch_text)
        decoder_states = [ntorch.zeros(init_state.shape["embedding"], 
                                       names=("hidden")).to(device)]
        log_probs = []
        for i in range(embedded.shape["trgSeqlen"]):
            last_hidden = decoder_states[-1]
            context = init_state.dot("hidden", last_hidden).softmax("srcSeqlen") \
                                .dot("srcSeqlen", init_state)
            curr_word = embedded[{"trgSeqlen": i}]
            rnn_input = ntorch.cat((curr_word, context), "embedding")
            output, hidden = self.lstm(rnn_input, last_hidden)
            decoder_states.append(hidden)
            log_probs = ntorch.cat((log_probs, self.w(output)), 2)
        
        return log_probs