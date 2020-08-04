"""
使用多层Linear()+Relu()进行10分类
"""

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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


examples = iter(train_loader)
samples, labels = examples.next()

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()



test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)



"""
model:
    input: X = X.view(-1, 784)
    
    l1 = torch.nn.Linear(784, 512)
    x = F.relu(l1(x))
    
    l2 = torch.nn.Linear(512, 256)
    x = F.relu(l2(x))
    
    l3 = torch.nn.Linear(256, 128)
    x = F.relu(l3(x))
    
    l4 = torch.nn.Linear(128, 64)
    x = F.relu(l4(x))
    
    10 classfier:
    l5 = torch.nn.Linear(64, 10)
"""

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
model = model.to(device)


criterion = torch.nn.CrossEntropyLoss()
# momentum 动量 提高优化性能
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 1):
        inputs, target = data

        inputs = inputs.to(device)
        target = target.to(device)


        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx%300 == 0:
            print('[%d, %5d] loss: %.3f' %(epoch+1, batch_idx, running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100*correct/total}%%')




if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()

