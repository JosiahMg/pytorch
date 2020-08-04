import torch

"""
Train a model to learn:
 "hello" -> "ohlol" 
"""

input_size = 4
hidden_size = 8
batch_size = 1
num_layers = 2
seq_len = 5
embedding_size = 10
num_class = 4

# vocabulary
idx2char = ['e', 'h', 'l', 'o']

x_data = [[1, 0, 2, 2, 3]]  # "hello"  shape=(batch, seqLen)
y_data = [3, 1, 2, 3, 2]  # "ohlol"


# Reshape the inputs to (seqLen, batchSize, inputSize)
inputs = torch.LongTensor(x_data)

labels = torch.LongTensor(y_data)


class RnnModel(torch.nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # (batch, seqLen, embeddingSize)
        x, _ = self.rnn(x, hidden)
        # input.x_shape = (batchsize, seqLen, hiddenSize)
        # output.x_shape = (batchsize, seqLen, num_class)
        x = self.fc(x)
        print(x.shape)
        return x.view(-1, num_class)


net = RnnModel()



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss=%.3f' % (epoch+1, loss.item()))


