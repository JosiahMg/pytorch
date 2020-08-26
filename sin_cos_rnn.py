import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


TIME_STEP = 10
INPUT_SIZE = 1
learning_rate = 0.001

#linspace 生成等差数列
steps = np.linspace(0, np.pi, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

# plt.plot(steps, y_np, 'r-', label='target(cos)')
# plt.plot(steps, x_np, 'b-', label='input(sin)')
# plt.legend(loc='best') # 图例
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # r_out.shape:seq_len,batch,hidden_size*num_direction(1,10,32)
        # h_state.shape:num_layers*num_direction,batch,hidden_size(1,？,32), ？与 batch_first相关
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        # outs = self.out(r_out)
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        # len(outs):r_out.size(1),即(batch,10)
        last_out = torch.stack(outs, dim=1) # last_out.shape:1,10,1
        # last_out = outs
        return last_out, h_state

rnn = RNN()
print(rnn)

criterion = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps) # x_np.shape: 10
    y_np = np.cos(steps) # y_np.shape: 10

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]) # x.shape: 1*10*1
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]) # y.shape: 1*10*1

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(.05)

plt.ioff()
plt.show()
