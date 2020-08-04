import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(-1, 1)


n_samples, n_features = X.shape
print(f'features={n_features}, number={n_samples}')

_, input_dim = X.shape
output_dim = 1

x_test = torch.tensor([5.])


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(input_dim, output_dim)
print(f'Prediction before training:f(5)={model(x_test).item():.3f}')

learning_rate = 0.8
epochs = 500

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y, y_pred)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch%10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {loss:.3f}')


print(f'Prediction before training:f(5)={model(x_test).item():.3f}')

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

