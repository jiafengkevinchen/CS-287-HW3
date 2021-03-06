# Torch
import torch
# Text text processing library and methods for pretrained word embeddings
from torchtext import data, datasets
# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn
from torch import nn
from namedtensor.text import NamedField

import getpass
if getpass.getuser() not in ['franciscorivera', 'jiafengchen', 'yufeng.ling']:
    from tensorboardcolab import TensorBoardColab
    from tensorboardX import SummaryWriter
    tbc = TensorBoardColab()
    writer = SummaryWriter(log_dir=tbc.get_writer().get_logdir())

import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = NamedField(names=('srcSeqlen',), tokenize=tokenize_de)
EN = NamedField(names=('trgSeqlen',), tokenize=tokenize_en,
                init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

MAX_LEN = 20
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                         len(vars(x)['trg']) <= MAX_LEN)
src = open("valid.src", "w")
trg = open("valid.trg", "w")
for example in val:
    print(" ".join(example.src), file=src)
    print(" ".join(example.trg), file=trg)
src.close()
trg.close()

MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)
# print(DE.vocab.freqs.most_common(10))
# print("Size of German vocab", len(DE.vocab))
# print(EN.vocab.freqs.most_common(10))
# print("Size of English vocab", len(EN.vocab))
# print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"]) #vocab index for <s>, </s>

BATCH_SIZE = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=BATCH_SIZE,
    device=device,
    repeat=False, sort_key=lambda x: len(x.src))


batch = next(iter(train_iter))

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")

pad_idx_EN = EN.vocab.stoi['<pad>']
V_EN = len(EN.vocab)
