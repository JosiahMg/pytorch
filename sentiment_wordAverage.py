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


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAVGModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.linear = nn.Linear(embedding_size, output_size)

    def forward(self, text):
        # text.shape:(seq_len, batchSize)
        # embedded.shape:(seq_len, batchSize, embedding_size)
        embedded = self.embed(text)
        # embedded.shape:(batchSize, seq_len)
        # embedded = embedded.transpose(1, 0)
        embedded = embedded.permute(1, 0, 2)
        # pooled.shape:(batchSize, embedding_size)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1), 1).squeeze()
        return self.linear(pooled)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = WordAVGModel(vocab_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE, output_size=OUTPUT_SIZE, pad_idx=PAD_IDX)

print(count_parameters(model))

pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
        torch.save(model.state_dict(), "model\wordavg-model.pth")

    print("Epoch", epoch, "Train loss", train_loss, "Train acc", train_acc)
    print("Epoch", epoch, "Valid loss", valid_loss, "Valid acc", valid_acc)


model.load_state_dict(torch.load("model\wordavg-model.pth"))

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


test_loss, test_acc = evaluate(model, test_iterator, criterion)
print('wordavg model test loss: ', test_loss, "accuracy:", test_acc)
