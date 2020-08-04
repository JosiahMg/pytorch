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

TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
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


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, num_filters, filter_size, dropout):
        super(CNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size))
        self.linear = nn.Linear(num_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)  # input_text.shape:(seq_len, batchSize) output_text.shape:(batchSize, seq_len)
        embedded = self.embed(text)  # embedded.shape:(batchSize, seq_len, embedding_size)
        embedded = embedded.unsqueeze(1)    # embedded.shape:(batchSize, 1, seqLen, embedding_size)
        conved = F.relu(self.conv(embedded))  # conved.shape:(batchSize, num_filters, seqLen-filter_size+1, 1)
        conved = conved.squeeze(3)      # conved.shape:(batchSize, num_filters, seqLen-filter_size+1)
        pooled = F.max_pool1d(conved, conved.shape[2])  # pooled.shape:(batchSize, num_filters,  1)
        pooled = pooled.squeeze(2)  # pooled.shape:(batchSize, num_filters)
        pooled = self.dropout(pooled)
        return self.linear(pooled)

# 多个covn并联组合的CNN，一般效果和一个covn差不多，需要其他方面的模型调试
class CNNsModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, num_filters, filter_sizes, dropout):
        super(CNNsModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        # self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size))
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_size))
            for fs in filter_sizes
        ]) # 多个cnn组合
        self.linear = nn.Linear(num_filters*len(filter_sizes), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)  # input_text.shape:(seq_len, batchSize) output_text.shape:(batchSize, seq_len)
        embedded = self.embed(text)  # embedded.shape:(batchSize, seq_len, embedding_size)
        embedded = embedded.unsqueeze(1)    # embedded.shape:(batchSize, 1, seqLen, embedding_size)
        # conved = F.relu(self.conv(embedded))  # conved.shape:(batchSize, num_filters, seqLen-filter_size+1, 1)
        # conved = conved.squeeze(3)      # conved.shape:(batchSize, num_filters, seqLen-filter_size+1)
        # pooled = F.max_pool1d(conved, conved.shape[2])  # pooled.shape:(batchSize, num_filters,  1)
        # pooled = pooled.squeeze(2)  # pooled.shape:(batchSize, num_filters)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, dim=1)  # pooled.shape:(batch, len(filter_sizes)*num_filters)
        pooled = self.dropout(pooled)
        return self.linear(pooled)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = CNNModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE,
                 output_size=OUTPUT_SIZE, pad_idx=PAD_IDX,
                 num_filters=100, filter_size=3, dropout=0.5)


# model = CNNsModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE,
#                  output_size=OUTPUT_SIZE, pad_idx=PAD_IDX,
#                  num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5)


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
        optimizer.step()

        epoch_loss += loss.item()*len(batch.label)
        epoch_acc += acc.item()*len(batch.label)
        total_len += len(batch.label)
    return epoch_loss/total_len, epoch_acc/total_len


def evaluate(model, iterator, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
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
        torch.save(model.state_dict(), "cnn-model.pth")

    print("Epoch", epoch, "Train loss", train_loss, "Train acc", train_acc)
    print("Epoch", epoch, "Valid loss", valid_loss, "Valid acc", valid_acc)



model.load_state_dict(torch.load("cnn-model.pth"))


test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('CNN model test loss: ', test_loss, "accuracy:", test_acc)



import spacy
nlp = spacy.load("en")

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    # tensor.shape:(seq_len, batch)
    tensor = tensor.unsqueeze(1)
    pred = torch.sigmoid(model(tensor))
    return pred.item()


print(predict_sentiment("This film is horrible!"))
print(predict_sentiment("This film is terrible!"))
print(predict_sentiment("This film is great!"))
print(predict_sentiment("This film is terrific!"))



