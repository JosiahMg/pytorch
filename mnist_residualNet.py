import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

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

test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.nn.functional.relu(y+x)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rbblock1 = ResidualBlock(16)
        self.rbblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.nn.functional.relu(self.mp(self.conv1(x)))
        x = self.rbblock1(x)
        x = torch.nn.functional.relu(self.mp(self.conv2(x)))
        x = self.rbblock2(x)
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
