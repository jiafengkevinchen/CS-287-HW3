import torch
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Beam(nnn.Module):
    def __init__(self, beam_size=3, variance=10):
        super().__init__()
        self.beam_size = beam_size
        self.variance = variance

    def forward(self, model, src):
        batch_size = src.shape['batch']
        nodes = (ntorch.ones((batch_size, self.beam_size, 1),
                             names=('batch', 'beam', 'trgSeqlen'))
                 * EN.vocab.stoi["<s>"]).long().to(device)
        scores = ntorch.zeros((batch_size, self.beam_size),
                              names=('batch', 'beam')).to(device)
        finished = [[] for _ in range(batch_size)]
        flag = [1] * batch_size
        sentence_length = 1
        while sum(flag) != 0 and sentence_length < 50:
            sentence_length += 1
            if sentence_length == 2:
                # start with SOS
                print(model(src, nodes[{'beam': 0}]))
                log_prob = model(src, nodes[{'beam': 0}])[{'trgSeqlen': -1}]
                vocab_size = log_prob.shape['classes']
            else:
                log_prob = ntorch.cat([model(src,
                                             nodes[{'beam': b}])[{'trgSeqlen': -1}]
                                       + scores[{'beam': b}]
                                       for b in range(self.beam_size)],
                                      'classes')
                vocab_size = int(log_prob.shape['classes'] / self.beam_size)
            best_scores, best_score_loc = log_prob.topk(
                'classes', self.variance)
            # advance trgSeqlen
            nodes = ntorch.cat([nodes, ntorch.ones((batch_size, self.beam_size, 1),
                                                   names=('batch', 'beam', 'trgSeqlen'))
                                .long().to(device)],
                               'trgSeqlen')
            for b in range(batch_size):
                beam_count = 0
                nodes_temp = ntorch.zeros((sentence_length, self.beam_size),
                                          names=('trgSeqlen', 'beam')).long().to(device)
                if len(finished[b]) < self.variance:
                    for k in range(self.variance):
                        prev = (
                            best_score_loc[{'batch': b, 'classes': k}] / vocab_size)
                        word = best_score_loc[{'batch': b, 'classes': k}].fmod(
                            vocab_size)
                        sentence = NamedTensor(torch.cat([nodes[{'batch': b, 'beam': prev}].values[:-1],
                                                          torch.LongTensor([word.values]).to(device)]),
                                               names=('trgSeqlen'))
                        if (word.values.item() == EN.vocab.stoi["</s>"] and
                                len(finished[b]) < self.variance):
                            finished[b].append(sentence)
                        else:
                            nodes_temp[{'beam': beam_count}] = sentence
                            scores[{'batch': b, 'beam': beam_count}] = best_scores[{
                                'batch': b, 'classes': k}].values.item()
                            beam_count += 1
                            if beam_count == self.beam_size:
                                break
                    nodes[{'batch': b}] = nodes_temp.to(device)
                else:
                    flag[b] = 0

        return finished
