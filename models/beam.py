import torch
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Beam(nnn.Module):
    def __init__(self, beam_size=3, variance=20):
        super().__init__()
        self.beam_size = beam_size
        self.variance = variance

    def forward(self, model, src):
        encoded = model.encoder(src)
        vocab_size = src.shape['classes']
        batch_size = src.shape['batch']
        nodes = ntorch.ones((batch_size, self.beam_size, 1),
                            names=('batch', 'beam', 'trgSeqlen')).to(device) \
                * EN.vocab.stoi["<s>"]
        scores = ntorch.zeros((batch_size, self.beam_size),
                                   names=('batch', 'beam'))
        finished = [[]] * batch_size
        sentence_length = 1
        while mask.sum().sum() != 0 and sentence_length < 100:
            sentence_length += 1
            log_prob = ntorch.cat([model(encoded, nodes[{'beam': b}])[{'trgSeqlen': -1}] + scores[{'beam': b}]
                                   for b in range(self.beam_size)],
                                  'classes')
            best_scores, best_score_loc = log_prob.topk('classes', self.variance)
            # advance trgSeqlen by 1
            nodes = torch.cat(nodes, ntorch.zeros((batch_size, self.beam_size, 1),
                                                  names=('batch', 'beam', 'trgSeqlen')).to(device),
                              'trgSeqlen')
            for b in range(batch_size):
                beam_count = 0
                nodes_temp = ntorch.zeros((sentence_length, beam_size),
                                          names=('trgSeqlen', 'beam')).to(device)
                if len(finished[b]) < self.variance:
                    for k in range(self.variance):
                        prev = (best_score_loc[{'batch': b, 'classes': k}] / vocab_size).floor()
                        word = best_score_loc[{'batch': b, 'classes': k}].fmod(vocab_size)
                        sentence = NamedTensor(torch.cat(nodes[{'batch': b, 'beam': prev}],
                                                         torch.FloatTensor([word])),
                                               names=('trgSeqlen')).to(device)
                        if word == EN.vocab.stoi["</s>"] and len(finished[b]) < self.variance:
                            finished[b].append(sentence)
                        else:
                            nodes_temp[{'beam': beam_count}] = sentence
                            score[{'batch': b, 'beam': beam_count}] = best_scores[{'batch': b, 'classes': k}]
                            beam_count += 1
                            if beam_count == beam_size:
                                break

        return finished


