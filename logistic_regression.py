import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 归一化数据集，会显著提供准确率
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print(input_dim, output_dim)


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(input_dim, output_dim)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_train, y_pred)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f'epoch = {epoch+1} loss = {loss.item():.4f}')


with torch.no_grad():
    y_test_pred = model(x_test)
    y_test_pred = y_test_pred.round()
    acc = y_test_pred.eq(y_test).sum().item() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
