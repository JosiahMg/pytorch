import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torchvision


import matplotlib.pyplot as plt
import time
import os
import copy


print("Torchvision Version: ", torchvision.__version__)

data_dir = "./data/hymenoptera_data"

model_name = "resnet"    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
num_classes = 2   # Number of classes in the dataset
batch_size = 32   # Batch size for training (change depending on how much memory you have)
num_epochs = 15   # Number of epochs to train for
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
input_size = 224


def image_normalize(dataset):
    n_channels = dataset[0][0].shape[0]
    for i in range(n_channels):
        data = [d[0][i].data.cpu().numpy() for d in dataset]
        print(f"train_data channnel[{i}] mean: {np.mean(data)}")
        print(f"train_data channnel[{i}] std: {np.std(data)}")




all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
        transforms.RandomResizedCrop(input_size),  # 安装尺寸input_size截取图片
        transforms.RandomHorizontalFlip(),         # 旋转图片，用于增加图片噪声
        transforms.ToTensor(),
    ]))
loader = torch.utils.data.DataLoader(all_imgs, batch_size=batch_size, shuffle=True)
print(len(all_imgs))
# image_normalize(all_imgs)


img = next(iter(loader))[0]
print(img.shape)


unloader = transforms.ToPILImage()
plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(2)  # pause a bit so that plots are updated


# plt.figure()
# imshow(img[0], title="Image_orginal")


data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),  # 安装尺寸input_size截取图片
        transforms.RandomHorizontalFlip(),         # 旋转图片，用于增加图片噪声
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
                  for x in ["train", "val"]}

dataloader_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ["train", "val"]}

img = next(iter(dataloader_dict['val']))[0]
print("image shape: ", img.shape)
# imshow(img[10], title="Image_normalize")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameters_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        # pretrained=True表示使用训练好的模型参数，否则随机生成参数的值
        model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
        set_parameters_requires_grad(model_ft, feature_extract)
        # print(model_ft.fc)
        num_ftrs = model_ft.fc.in_features
        # 修改原先Linear的feature值
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Model not implemented")
        return None, None

    return model_ft, input_size

# 使用预训练模型，只训练最后一层fc的参数
model_ft, input_size = initialize_model(model_name, num_classes,
                                        feature_extract=True, use_pretrained=True)

print(model_ft.fc.weight.requires_grad)
print(model_ft.layer1[0].conv1.weight.requires_grad)

model_ft = model_ft.to(device)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model_ft.parameters()),
    lr=0.001, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_loss_history = []
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            for inputs, labels in dataloader_dict[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                preds = outputs.argmax(dim=1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print(f"Phase {phase} loss: {epoch_loss}, acc: {epoch_acc}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "val":
                val_loss_history.append(epoch_acc)

    model.load_state_dict(best_model_wts)
    return model, val_loss_history


_, ohist = train_model(model_ft, dataloader_dict, loss_fn, optimizer, num_epochs=num_epochs)


# 不使用预训练模型，所有参数自己训练
model_scratch, _ = initialize_model(model_name, num_classes,
                                    feature_extract=False, use_pretrained=False)

model_scratch = model_scratch.to(device)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model_scratch.parameters()),
    lr=0.001, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()
_, scratch_hist = train_model(model_scratch, dataloader_dict, loss_fn, optimizer, num_epochs=num_epochs)


plt.title("Validation Accuracy vs. Number of Trainning Epochs")
plt.title("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, num_epochs+1), ohist, label="Pretrained")
plt.plot(range(1, num_epochs+1), scratch_hist, label="Scratch")
plt.ylim(0, 1.)
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
plt.pause(10)