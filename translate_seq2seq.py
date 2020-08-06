import torchvision
import torch
import os
import sys
from collections import Counter
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import nltk
import jieba


def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("\t")
            en.append(['BOS'] + nltk.word_tokenize(line[0].lower()) + ['EOS'])
            # cn.append(['BOS'] + [c for c in line[1]] + ['EOS'])
            cn.append(['BOS'] + jieba.lcut(line[1]) + ['EOS'])
    return en, cn


train_file = 'data/translate_train.txt'
dev_file = 'data/translate_dev.txt'
train_en, train_cn = load_data(train_file)
dev_en, dev_cn = load_data(dev_file)

UNK_IDX = 0
PAD_IDX = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 2
    word_dict = {w[0]: index for index, w in enumerate(ls, 2)}
    word_dict['UNK'] = UNK_IDX
    word_dict['PAD'] = PAD_IDX
    return word_dict, total_words


en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)

inv_en_dict = {v: k for k, v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}


# 把句子变成索引
def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    """
    Encode the sequences.
    """
    length = len(en_sentences)
    # 将句子的词转换成词典对应的索引
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]

    return out_en_sentences, out_cn_sentences


train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)

k = 10000
print(" ".join([inv_cn_dict[i] for i in train_cn[k]]))
print(" ".join([inv_en_dict[i] for i in train_en[k]]))


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size) # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths #x_mask

def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex

batch_size = 64
train_data = gen_examples(train_en, train_cn, batch_size)
random.shuffle(train_data)
dev_data = gen_examples(dev_en, dev_cn, batch_size)


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx]
        embedded = self.dropout(self.embed(x_sorted))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        pack_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)

        _, original_idx = sorted_idx.sort(0, descending=True)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        return out, hid[[-1]]


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, lengths, hid):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx]
        hid = hid[:, sorted_idx]  # C from encoder
        embedded = self.dropout(self.embed(y_sorted))  # embedded.shape:(batch, seqLen, hidden_size)

        # embedded.shape:(batch, seqLen, 2*hidden_size)
        # concat C into inputs
        # embedded = torch.cat([embedded, hid.unsqueeze(0).expand_as(embedded)], dim=2)
        embedded = torch.cat([embedded, hid.transpose(1, 0).expand_as(embedded)], dim=2)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        pack_out, newhid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)

        _, original_idx = sorted_idx.sort(0, descending=True)
        out = out[original_idx.long()].contiguous()  # out.shape:(batch, seqLen, hidden_size)
        newhid = newhid[:, original_idx.long()].contiguous()  # hid.shape:(1, batch, hidden_size)

        # embedded.shape:(batch, seqLen, 2*hidden_size)
        # concat C into hidden
        # hid.transpose(1, 0).expand_as(out) 的功能如下：
        # (1, batchSize, hidden) -> (batchSize, 1, hidden) -> (batchSize, seqLen, hidden)
        # out = torch.cat([out, hid.unsqueeze(0).expand_as(out)], dim=2)
        out = torch.cat([out, hid.transpose(1, 0).expand_as(out)], dim=2)
        out = self.fc(out)  # out.shape:(batch, seqLen, vocab_size)

        out = F.log_softmax(out, -1)
        return out, newhid


class PlainSeq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_len, y, y_len):
        encoder_out, hid = self.encoder(x, x_len)
        output, hid = self.decoder(y, y_len, hid)
        return output, None

    def translate(self, x, x_lengths, y, max_length=10): #x.shape=(1, seqLen)
        encoder_out, hid = self.encoder(x, x_lengths)  #hid.shape:(1, 1, hiddenSize)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid = self.decoder(y, torch.ones(batch_size).long().to(y.device), hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)

        return torch.cat(preds, 1), None


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()



    def forward(self, input, target, mask):
        input = input.contiguous().view(-1, input.size(2))  # input.shape=(batch, seqLen, vocabSize)
        target = target.contiguous().view(-1, 1)   # input_target.shape(batch, seqLen)
        mask = mask.contiguous().view(-1, 1)    # input_mask.shape:(batch,seqLen)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


dropout = 0.2
hidden_size = 100
encoder = PlainEncoder(vocab_size=en_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
decoder = PlainDecoder(vocab_size=cn_total_words,
                      hidden_size=hidden_size,
                      dropout=dropout)
model = PlainSeq2seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())



def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)


def train(model, data, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()   # shape:(batch, seqLen)
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()  # shape:(batch)
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()  # shape:(batch, seqLen)
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()  # shape:(batch, seqLen)
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)  # mb_pred.shape:(batch, seqLen, vocabSize)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()  # mb_out_mask.shape=(batch, seqLen)

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch", epoch, "iteration", it, "loss", loss.item())

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)


train(model, train_data, num_epochs=20)

def translate_dev(i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))

for i in range(100,120):
    translate_dev(i)
    print()

