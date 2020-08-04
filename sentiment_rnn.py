import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import random
import torch.nn.functional as F

SEED = 1234
BATCH_SIZE = 64
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
NUM_EPOCHS = 10
GRAD_CLIP = 5


torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(root='data', text_field=TEXT, label_field=LABEL)

print(f'Number of trainning exammples: {len(train_data)}')
print(f'Number of testing exammples: {len(test_data)}')

print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f'Number of trainning exammples: {len(train_data)}')
print(f'Number of validation exammples: {len(valid_data)}')
print(f'Number of testing exammples: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=20000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

VOCAB_SIZE = len(TEXT.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=BATCH_SIZE, device=device)


batch = next(iter(valid_iterator))
print(batch.text.shape)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout, bidirectional=True, num_layers=1):
        super(RNNModel, self).__init__()
        self.n_directions = 2 if bidirectional else 1
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size*self.n_directions, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text.shape:(seq_len, batchSize)
        # embedded.shape:(seq_len, batchSize, embedding_size)
        embedded = self.embed(text)
        embedded = self.dropout(embedded)
        # hidden.shape:(n_direction*n_layers, batchSize, hiddenSize)
        _, (hidden, _) = self.lstm(embedded)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        hidden_cat = self.dropout(hidden_cat)
        return self.linear(hidden_cat)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = RNNModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE,
                 output_size=OUTPUT_SIZE, pad_idx=PAD_IDX,
                 hidden_size=100, dropout=0.5)

print(count_parameters(model))

pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.train()
    total_len = 0.
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()*len(batch.label)
        epoch_acc += acc.item()*len(batch.label)
        total_len += len(batch.label)
    return epoch_loss/total_len, epoch_acc/total_len


def evaluate(model, iterator, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
    with torch.no_grad():
        for batch in iterator:
            preds = model(batch.text).squeeze()
            loss = crit(preds, batch.label)
            acc = binary_accuracy(preds, batch.label)

            epoch_loss += loss.item()*len(batch.label)
            epoch_acc += acc.item()*len(batch.label)
            total_len += len(batch.label)

    model.train()
    return epoch_loss/total_len, epoch_acc/total_len


best_valid_acc = 0.
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), "lstm-model.pth")

    print("Epoch", epoch, "Train loss", train_loss, "Train acc", train_acc)
    print("Epoch", epoch, "Valid loss", valid_loss, "Valid acc", valid_acc)


test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('RNN model test loss: ', test_loss, "accuracy:", test_acc)