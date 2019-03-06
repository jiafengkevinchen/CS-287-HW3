import torch
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

class Beam(nnn.Module):
    def __init__(self, beam_size=3, topk=10, max_len=50):
        super().__init__()
        self.beam_size = beam_size
        self.topk = topk
        self.max_len = max_len
        self.scores = None
        self.nodes = None
        self.result = None
        self.filled = None
        self.top_scores = None
        self.top_score_locs = None

    def log_prob(self, model, src):
        # calculate marginal probability for next word
        if nodes.shape['trgSeqlen'] == 1:
            # start with SOS
            log_prob = model(src, self.nodes[{'beam': 0}])[{'trgSeqlen': -1}]
            
            return log_prob, log_prob.shape['classes']
        else:
            log_prob = [model(src, nodes[{'beam': b}])[{'trgSeqlen': -1}]
                        + self.scores[{'beam': b}]
                        for b in range(self.beam_size)]
            return ntorch.cat(log_prob,'classes'), log_prob.shape['classes']

    def generate_sentence(self, b, k, vocab_size):
        sentence_i = self.top_score_loc[{'batch': b, 'classes': k}] / vocab_size
        prev_sentence = self.nodes[{'batch': b, 'beam': sentence_i}].values[:-1]
        word = self.top_score_locs[{'batch': b, 'classes': k}].fmod(vocab_size).long()
        sentence = NamedTensor(torch.cat([prev_sentence, word], names=('trgSeqlen')))

        return sentence

    def advance(self, batch_size, vocab_size):
        # increment length and pad using 1
        increment = ntorch.ones((batch_size, self.beam_size, 1),
                                names=('batch', 'beam', 'trgSeqlen')) \
                    .long().to(device)
        self.nodes = ntorch.cat([nodes, increment], 'trgSeqlen')

        # generate new next word for the entire batch
        for b in range(batch_size):
            beam_count = 0
            new_nodes = ntorch.zeros((self.nodes.shape['trgSeqlen'] + 1,
                                      self.beam_size),
                                     names=('trgSeqlen', 'beam')) \
                        .long().to(device)
            if len(self.result[b]) < self.topk:
                for k in range(self.topk):
                    sentence = self.generate_sentence(b, k, vocab_size)
                    added_word = sentence[{'trgSeqlen': -1}]
                    if len(self.result[b]) < self.topk and \
                        (added_word.values.item() == EN.vocab.stoi["</s>"]
                         or self.nodes.shape['trgSeqlen'] == self.max_len):
                        self.result[b].append(sentence)
                    else:
                        new_nodes[{'beam': beam_count}] = sentence
                        self.scores[{'batch': b, 'beam': beam_count}] = \
                            self.top_scores[{'batch': b, 'classes': k}].values.item()
                        beam_count += 1
                        if beam_count == self.beam_size:
                            break
                self.nodes[{'batch': b}] = nodes_temp.to(device)
            else:
                self.filled[b] = True

    def forward(self, model, src):
        batch_size = src.shape['batch']
        self.nodes = (ntorch.ones((batch_size, self.beam_size, 1),
                                  names=('batch', 'beam', 'trgSeqlen'))
                     * EN.vocab.stoi["<s>"]).long().to(device)
        self.scores = ntorch.zeros((batch_size, self.beam_size),
                                   names=('batch', 'beam')).to(device)
        self.result = [[] for _ in range(batch_size)]
        self.filled = [False] * batch_size
        
        while sum(flag) < batch_size and self.nodes.shape['trgSeqlen'] <= self.max_len:
            log_prob, vocab_size = self.log_prob(model, src)
            self.top_scores, self.top_score_locs = log_prob.topk('classes', self.topk)
            self.advance(batch_size, vocab_size)

        return self.result
