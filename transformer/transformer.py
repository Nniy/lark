# Standard PyTorch imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import spacy
from torch.autograd import Variable
from torchtext import data, datasets

# For plots
import matplotlib.pyplot as plt


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return output


def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=0.0):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code. Initialize parameters with Glorot or fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class NoamOpt:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


crit = LabelSmoothing(5, 0, 0.2)


def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
    print(predict)
    return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()


def loss_backprop(generator, criterion, out, targets, normalize):

    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.item()
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total


def make_std_mask(src, tgt, pad):
    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask


def train_epoch(train_iter, model, criterion, opt, transpose=False):
    model.train()
    for i, batch in enumerate(train_iter):
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)

        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, model_opt._rate)


def valid_epoch(valid_iter, model, criterion, transpose=False):
    model.test()
    total = 0
    for batch in valid_iter:
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        loss = loss_backprop(model.generator, criterion, out, trg[:, 1:], batch.ntokens)


class Batch:
    def __init__(self, src, trg, src_mask, trg_mask, ntokens):
        self.src = src
        self.trg = trg
        self.src_mask = src_mask
        self.trg_mask = trg_mask
        self.ntokens = ntokens


def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        src_mask, tgt_mask = make_std_mask(src, tgt, 0)
        yield Batch(src, tgt, src_mask, tgt_mask, (tgt[1:] != 0).data.sum())


V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = get_std_opt(model)
for epoch in range(2):
    train_epoch(data_gen(V, 30, 20), model, criterion, model_opt)
#
# spacy_de = spacy.load('de')
# spacy_en = spacy.load('en')
#
#
# def tokenize_de(text):
#     return [tok.text for tok in spacy_de.tokenizer(text)]
#
#
# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(text)]
#
#
# BOS_WORD = '<s>'
# EOS_WORD = '</s>'
# BLANK_WORD = "<blank>"
# SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
# TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
#                  eos_token=EOS_WORD, pad_token=BLANK_WORD)
#
# MAX_LEN = 100
# train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
#                                          filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
#                                                                len(vars(x)['trg']) <= MAX_LEN)
# MIN_FREQ = 1
# SRC.build_vocab(train.src, min_freq=MIN_FREQ)
# TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
#
# # Detail. Batching seems to matter quite a bit.
# # This is temporary code for dynamic batching based on number of tokens.
# # This code should all go away once things get merged in this library.
#
# BATCH_SIZE = 4096
# global max_src_in_batch, max_tgt_in_batch
#
#
# def batch_size_fn(new, count, sofar):
#     "Keep augmenting batch and calculate total number of tokens + padding."
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)
#
#
# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(
#                         sorted(p, key=self.sort_key),
#                         self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#
#             self.batches = pool(self.data(), self.random_shuffler)
#
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                 self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))
#
#
# def rebatch(pad_idx, batch):
#     "Fix order in torchtext to match ours"
#     src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
#     src_mask, trg_mask = make_std_mask(src, trg, pad_idx)
#     return Batch(src, trg, src_mask, trg_mask, (trg[1:] != pad_idx).data.sum())
#
#
# train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=True)
# valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=False)
#
# pad_idx = TGT.vocab.stoi["<blank>"]
# model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
# model_opt = get_std_opt(model)
# model.cuda()
#
# criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
# criterion.cuda()
# for epoch in range(15):
#     train_epoch((rebatch(pad_idx, b) for b in train_iter), model, criterion, model_opt)
#     valid_epoch((rebatch(pad_idx, b) for b in valid_iter), model, criterion)







# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
#
# class Batch:
#     """Object for holding a batch of data with mask during training."""
#
#     def __init__(self, src, trg=None, pad=0):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         if trg is not None:
#             self.trg = trg[:, :-1]
#             self.trg_y = trg[:, 1:]
#             self.trg_mask = self.make_std_mask(self.trg, pad)
#             self.ntokens = (self.trg_y != pad).data.sum()
#
#     @staticmethod
#     def make_std_mask(tgt, pad):
#         """Create a mask to hide padding and future words."""
#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
#         return tgt_mask
#
#
# def run_epoch(data_iter, model, loss_compute):
#     """Standard Training and Logging Function"""
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     for i, batch in enumerate(data_iter):
#         out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
#         loss = loss_compute(out, batch.trg_y, batch.ntokens)
#         total_loss += loss
#         total_tokens += batch.ntokens
#         tokens += batch.ntokens
#         if i % 50 == 1:
#             elapsed = time.time() - start
#             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
#             start = time.time()
#             tokens = 0
#     return total_loss / total_tokens
#
#
# global max_src_in_batch, max_tgt_in_batch
#
#
# def batch_size_fn(new, count, sofar):
#     """Keep augmenting batch and calculate total number of tokens + padding."""
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)
#
#
# class NoamOpt:
#     """Optim wrapper that implements rate."""
#
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0
#
#     def step(self):
#         """Update parameters and rate"""
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
#
#     def rate(self, step=None):
#         """Implement `lrate` above"""
#         if step is None:
#             step = self._step
#         return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
#
#
# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
#
# # opts = [NoamOpt(512, 1, 4000, None),
# # #         NoamOpt(512, 1, 8000, None),
# # #         NoamOpt(256, 1, 4000, None)]
# # # plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# # # plt.legend(["512:4000", "512:8000", "256:4000"])
# # # plt.show()
#
#
# class LabelSmoothing(nn.Module):
#     """Implement label smoothing."""
#
#     def __init__(self, size, padding_idx, smoothing=0.0):
#         super(LabelSmoothing, self).__init__()
#         self.criterion = nn.KLDivLoss(size_average=False)
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None
#
#     def forward(self, x, target):
#         assert x.size(1) == self.size
#         true_dist = x.data.clone()
#         true_dist.fill_(self.smoothing / (self.size - 2))
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         true_dist[:, self.padding_idx] = 0
#         mask = torch.nonzero(target.data == self.padding_idx)
#         if mask.dim() > 0:
#             true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         return self.criterion(x, Variable(true_dist, requires_grad=False))
#
#
# # # Example of label smoothing.
# # crit = LabelSmoothing(5, 0, 0.4)
# # predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
# #                              [0, 0.2, 0.7, 0.1, 0],
# #                              [0, 0.2, 0.7, 0.1, 0]])
# # v = crit(Variable(predict.log()),
# #          Variable(torch.LongTensor([2, 1, 0])))
# #
# # # Show the target distributions expected by the system.
# # plt.imshow(crit.true_dist)
# # plt.show()
#
# crit = LabelSmoothing(5, 0, 0.1)
#
#
# def loss(x):
#     d = x + 3 * 1
#     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
#     # print(predict)
#     return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()  # was .data[0]
#

# plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
# plt.show()


# # First example
#
# def data_gen(V, batch, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         data[:, 0] = 1
#         src = Variable(data, requires_grad=False)
#         tgt = Variable(data, requires_grad=False)
#         yield Batch(src, tgt, 0)
#
#
# class SimpleLossCompute:
#     "A simple loss compute and train function."
#
#     def __init__(self, generator, criterion, opt=None):
#         self.generator = generator
#         self.criterion = criterion
#         self.opt = opt
#
#     def __call__(self, x, y, norm):
#         x = self.generator(x)
#         loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
#                               y.contiguous().view(-1)) / norm
#         loss.backward()
#         if self.opt is not None:
#             self.opt.step()
#             self.opt.optimizer.zero_grad()
#         return loss.item() * norm  # was .data[0]
#
#
# # Train the simple copy task.
# V = 11
# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# model = make_model(V, V, N=2)
# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
#                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#
# for epoch in range(10):
#     model.train()
#     run_epoch(data_gen(V, 30, 20), model,
#               SimpleLossCompute(model.generator, criterion, model_opt))
#     model.eval()
#     print(run_epoch(data_gen(V, 30, 5), model,
#                     SimpleLossCompute(model.generator, criterion, None)))
#
#
# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
#     for i in range(max_len - 1):
#         out = model.decode(memory, src_mask,
#                            Variable(ys),
#                            Variable(subsequent_mask(ys.size(1))
#                                     .type_as(src.data)))
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()  # was .data[0]
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#     return ys
#
#
# model.eval()
# src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
# src_mask = Variable(torch.ones(1, 1, 10))
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

# # For data loading.
# from torchtext import data, datasets
#
# if True:
#     import spacy
#
#     spacy_de = spacy.load('de')
#     spacy_en = spacy.load('en')
#
#
#     def tokenize_de(text):
#         return [tok.text for tok in spacy_de.tokenizer(text)]
#
#
#     def tokenize_en(text):
#         return [tok.text for tok in spacy_en.tokenizer(text)]
#
#
#     BOS_WORD = '<s>'
#     EOS_WORD = '</s>'
#     BLANK_WORD = "<blank>"
#     SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
#     TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
#
#     MAX_LEN = 100
#     train, val, test = datasets.IWSLT.splits(
#         exts=('.de', '.en'), fields=(SRC, TGT),
#         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
#     MIN_FREQ = 2
#     SRC.build_vocab(train.src, min_freq=MIN_FREQ)
#     TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
#
#
# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#
#             self.batches = pool(self.data(), self.random_shuffler)
#
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))
#
#
# def rebatch(pad_idx, batch):
#     """Fix order in torchtext to match ours"""
#     src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
#     return Batch(src, trg, pad_idx)
#
#
# # Skip if not interested in multigpu.
# class MultiGPULossCompute:
#     """A multi-gpu loss compute and train function."""
#
#     def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
#         # Send out to different gpus.
#         self.generator = generator
#         self.criterion = nn.parallel.replicate(criterion,
#                                                devices=devices)
#         self.opt = opt
#         self.devices = devices
#         self.chunk_size = chunk_size
#
#     def __call__(self, out, targets, normalize):
#         total = 0.0
#         generator = nn.parallel.replicate(self.generator,
#                                           devices=self.devices)
#         out_scatter = nn.parallel.scatter(out,
#                                           target_gpus=self.devices)
#         out_grad = [[] for _ in out_scatter]
#         targets = nn.parallel.scatter(targets,
#                                       target_gpus=self.devices)
#
#         # Divide generating into chunks.
#         chunk_size = self.chunk_size
#         for i in range(0, out_scatter[0].size(1), chunk_size):
#             # Predict distributions
#             out_column = [[Variable(o[:, i:i + chunk_size].data,
#                                     requires_grad=self.opt is not None)]
#                           for o in out_scatter]
#             gen = nn.parallel.parallel_apply(generator, out_column)
#
#             # Compute loss.
#             y = [(g.contiguous().view(-1, g.size(-1)),
#                   t[:, i:i + chunk_size].contiguous().view(-1))
#                  for g, t in zip(gen, targets)]
#             loss = nn.parallel.parallel_apply(self.criterion, y)
#
#             # Sum and normalize loss
#             l = nn.parallel.gather(loss,
#                                    target_device=self.devices[0])
#             l = l.sum()[0] / normalize
#             total += l.data[0]
#
#             # Backprop loss to output of transformer
#             if self.opt is not None:
#                 l.backward()
#                 for j, l in enumerate(loss):
#                     out_grad[j].append(out_column[j][0].grad.data.clone())
#
#         # Backprop all loss through transformer.
#         if self.opt is not None:
#             out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
#             o1 = out
#             o2 = nn.parallel.gather(out_grad,
#                                     target_device=self.devices[0])
#             o1.backward(gradient=o2)
#             self.opt.step()
#             self.opt.optimizer.zero_grad()
#         return total * normalize
#
#
# # GPUs to use
# devices = [0]  # [0, 1, 2, 3]
# if True:
#     pad_idx = TGT.vocab.stoi["<blank>"]
#     model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
#     model.cuda()
#     criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
#     criterion.cuda()
#     BATCH_SIZE = 12000
#     train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
#                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                             batch_size_fn=batch_size_fn, train=True)
#     valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
#                             repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                             batch_size_fn=batch_size_fn, train=False)
#     model_par = nn.DataParallel(model, device_ids=devices)
#
# if False:
#     model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
#                         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
#     for epoch in range(10):
#         model_par.train()
#         run_epoch((rebatch(pad_idx, b) for b in train_iter),
#                   model_par,
#                   MultiGPULossCompute(model.generator, criterion,
#                                       devices=devices, opt=model_opt))
#         model_par.eval()
#         loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
#                          model_par,
#                          MultiGPULossCompute(model.generator, criterion,
#                                              devices=devices, opt=None))
#         print(loss)
# else:
#     model = torch.load("iwslt.pt")
#
# for i, batch in enumerate(valid_iter):
#     src = batch.src.transpose(0, 1)[:1]
#     src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
#     out = greedy_decode(model, src, src_mask,
#                         max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
#     print("Translation:", end="\t")
#     for i in range(1, out.size(1)):
#         sym = TGT.vocab.itos[out[0, i]]
#         if sym == "</s>": break
#         print(sym, end=" ")
#     print()
#     print("Target:", end="\t")
#     for i in range(1, batch.trg.size(0)):
#         sym = TGT.vocab.itos[batch.trg.data[i, 0]]
#         if sym == "</s>": break
#         print(sym, end=" ")
#     print()
#     break
#
# model, SRC, TGT = torch.load("en-de-model.pt")
#
# model.eval()
# sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
# src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
# src = Variable(src)
# src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
# out = greedy_decode(model, src, src_mask,
#                     max_len=60, start_symbol=TGT.stoi["<s>"])
# print("Translation:", end="\t")
# trans = "<s> "
# for i in range(1, out.size(1)):
#     sym = TGT.itos[out[0, i]]
#     if sym == "</s>": break
#     trans += sym + " "
# print(trans)
#
# tgt_sent = trans.split()
#
#
# def draw(data, x, y, ax):
#     seaborn.heatmap(data,
#                     xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
#                     cbar=False, ax=ax)
#
#
# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1, 4, figsize=(20, 10))
#     print("Encoder Layer", layer + 1)
#     for h in range(4):
#         draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
#              sent, sent if h == 0 else [], ax=axs[h])
#     plt.show()
#
# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1, 4, figsize=(20, 10))
#     print("Decoder Self Layer", layer + 1)
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
#              tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
#     plt.show()
#     print("Decoder Src Layer", layer + 1)
#     fig, axs = plt.subplots(1, 4, figsize=(20, 10))
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
#              sent, tgt_sent if h == 0 else [], ax=axs[h])
#     plt.show()
