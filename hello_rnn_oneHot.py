import torch

"""
Train a model to learn:
 "hello" -> "ohlol" 
"""

input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1
seq_len = 5

# vocabulary
idx2char = ['e', 'h', 'l', 'o']

x_data = [1, 0, 2, 2, 3]  # "hello"
y_data = [3, 1, 2, 3, 2]  # "ohlol"

one_hot_loopup = [[1, 0, 0, 0],     #'e'
                  [0, 1, 0, 0],     #'h'
                  [0, 0, 1, 0],     #'l'
                  [0, 0, 0, 1]]     #'o'

x_one_hot = [one_hot_loopup[x] for x in x_data]

# Reshape the inputs to (seqLen, batchSize, inputSize)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)

labels = torch.LongTensor(y_data)


class RnnModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RnnModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


net = RnnModel(input_size, hidden_size, batch_size)



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


