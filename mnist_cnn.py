import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    # mean:0.1307, std:0.3081
    # pixel(norm) = (pixel(origin）- mean )/std
    transforms.Normalize((0.1307, ), (0.3081, ))
])

# return.shape (tensor(X), y)   X.shape=1, 28, 28   y.shape=(1, )
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


tmp_data = [d[0].data.cpu().numpy() for d in train_dataset]
print("mean: ", np.mean(tmp_data))
print("std: ", np.std(tmp_data))


test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # x.shape:(batchSize, channels, height, width)
        batch_size = x.size(0)
        x = torch.relu(self.pooling(self.conv1(x)))
        x = torch.relu(self.pooling(self.conv2(x)))
        # flatten
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            print('[%d, %5d] loss: %.3f' %(epoch+1, batch_idx+1, running_loss/2000))
            running_loss = 0.0



def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Accuracy on test set: {100 * correct / total}%%')


if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()
