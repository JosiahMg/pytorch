import torch
import torch.nn as nn
from torchtext.vocab import Vectors
import torchtext
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)

if USE_CUDA:
    torch.cuda.manual_seed(53113)

device = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 50000
NUM_EPOCHS = 1
GRAD_CLIP = 5.0


TEXT = torchtext.data.Field(lower=True)

train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path="data", train="text8.train.txt",
                                                                     validation="text8.dev.txt", test="text8.test.txt",
                                                                     text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

print(TEXT.vocab.itos[:10])
print(TEXT.vocab.stoi["apple"])

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=BATCH_SIZE,
                                                                     device=device, bptt_len=50, repeat=False, shuffle=True)

it = iter(train_iter)
batch = next(it)

# batch.text.shape = (bptt_len, batch_size)
print(batch.text.shape)

print(" ".join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
print(" ".join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))

VOCAB_SIZE = len(TEXT.vocab)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, text, hidden):
        # text.shape: (seq_len, batch)
        # emb.shape: (seq_len, batch, embed_size)
        emb = self.embed(text)

        # output.shape: (seq_len, batch, hidden_size)
        # hidden.shape: (num_layers, batch, hidden_size)
        output, hidden = self.lstm(emb, hidden)
        # output = output.view(-1, output.shape[2])
        out_vocab = self.linear(output.view(-1, output.shape[2]))
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1))
        return out_vocab, hidden

    def init_hidden(self, batchSize, requries_grad=True):
        weight = next(self.parameters())
        return (weight.new_zeros((1, batchSize, self.hidden_size), requires_grad=requries_grad),
                weight.new_zeros((1, batchSize, self.hidden_size), requires_grad=requries_grad))



model = RNNModel(vocab_size=len(TEXT.vocab), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)

if USE_CUDA:
    model = model.to(device)



print(model)
print(next(model.parameters()))


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

val_losses = []


def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_loss = loss.item() * np.multiply(*data.size())
            total_count = np.multiply(*data.size())

    loss = total_loss/total_count
    model.train()
    return loss


for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        # output.shape:(batch, class_dim)
        # target.shape:(batch)
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if i% 100 == 0:
            print("epoch", epoch, "iteration", i, "loss", loss.item())

        # Save model
        if i%10000 == 0:
            val_loss = evaluate(model, val_iter)
            print("epoch", epoch, "iteration", i, "validation loss", val_loss)
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), "model\lm.pth")
                print('best model saved to lm.pth')
            else:
                # learning_rate decay
                scheduler.step()
                print('learning rate decay')
            val_losses.append(val_loss)



# Load model
best_model = RNNModel(vocab_size=len(TEXT.vocab), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)

if USE_CUDA:
    best_model = best_model.to(device)

best_model.load_state_dict(torch.load("model\lm.pth"))


# preplexity
val_loss = evaluate(model, val_iter)
print("perplexity: ", np.exp(val_loss))


# create sentences

hidden = best_model.init_hidden(1)
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    # softmax()
    word_weights = output.squeeze().exp().cpu()
    # greedy (argmax) eq torch.max()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))








