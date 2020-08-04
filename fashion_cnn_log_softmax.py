
"""
常用的ConvNet架构
INPUT -> [[CONV -> RELU] * N -> POOL] * M -> [FC -> RELU] * K -> FC
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


print("PyTorch Version: ", torch.__version__)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  # x.shape:(batchSize, 1, h, w)
        x = F.relu(self.conv1(x))  # x.shape:(batchSize, 20, 24, 24)
        x = F.max_pool2d(x, 2, 2)  # x.shape:(batchSize, 20, 12, 12)
        x = F.relu(self.conv2(x))  # x.shape:(batchSize, 50, 8, 8)
        x = F.max_pool2d(x, 2, 2)  # x.shape:(batchSize, 50, 4, 4)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))   # x.shape:(batchSize, 500)
        x = self.fc2(x)           # x.shape:(batchSize, 10)
        return F.log_softmax(x, dim=1)


train_data = datasets.FashionMNIST(root='data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.286,), (0.353,))
                            ]))

test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.286,), (0.353,))
                            ]))


def image_normalize(dataset):
    n_channels = dataset[0][0].shape[0]
    for i in range(n_channels):
        data = [d[0][i].data.cpu().numpy() for d in dataset]
        print(f"train_data channnel[{i}] mean: {np.mean(data)}")
        print(f"train_data channnel[{i}] std: {np.std(data)}")


image_normalize(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

lr = 0.01
momentum = 0.5
model = ConvNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data) # pred.shape:(batchSize, 10)
        loss = F.nll_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, idx, loss.item()))


def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)     # output.shape:(batchSize, 10)
            pred = output.argmax(dim=1)  # pred.shape:(batchSize, 1)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += pred.eq(target.view_as(target)).sum()

    total_loss /= len(test_loader.dataset)
    acc = correct/len(test_loader.dataset)*100
    print(f"Test loss: {total_loss}, Accuracy: {acc}")



num_opochs = 2
for epoch in range(num_opochs):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

torch.save(model.state_dict(), "fashion_cnn_log_softmax.pth")
