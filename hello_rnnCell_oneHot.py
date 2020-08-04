import torch

"""
Train a model to learn:
 "hello" -> "ohlol" 
"""

input_size = 4
hidden_size = 4
batch_size = 1

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
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)

# Reshape the labels to (seqLen, 1)
labels = torch.LongTensor(y_data).view(-1, 1)


class RnnCellModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(RnnCellModel, self).__init__()
        # self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    # Create h0
    def init_hidden0(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = RnnCellModel(input_size, hidden_size, batch_size)



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden0()
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')

    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))


