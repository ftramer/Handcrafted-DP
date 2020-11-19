"""
download model from https://github.com/bearpaw/pytorch-classification
"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from transfer.resnext import resnext
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR100(root='.data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

testset = datasets.CIFAR100(root='.data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

model = resnext(
    cardinality=8,
    num_classes=100,
    depth=29,
    widen_factor=4,
    dropRate=0,
)

model = torch.nn.DataParallel(model).cuda()
model.eval()

checkpoint = torch.load("transfer/resnext-8x64d/model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda()
        outputs = torch.argmax(model(inputs), dim=-1)

        acc += torch.sum(outputs.cpu().eq(targets))

    acc /= (1.0 * len(testset))
    acc = (100 * acc).numpy()
    print(f"Test Acc on CIFAR 100 = {acc: .2f}")

model.module.classifier = torch.nn.Identity()

features_cifar100_train = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_cifar100_train.append(f)

features_cifar100_train = np.concatenate(features_cifar100_train, axis=0)
print(features_cifar100_train.shape)

mean_cifar100 = np.mean(features_cifar100_train, axis=0)
var_cifar100 = np.var(features_cifar100_train, axis=0)

trainset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

testset = datasets.CIFAR10(root='.data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

ytrain = np.asarray(trainset.targets).reshape(-1)
ytest = np.asarray(testset.targets).reshape(-1)

features_train = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_train.append(f)

features_train = np.concatenate(features_train, axis=0)
print(features_train.shape)

features_test = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_test.append(f)

features_test = np.concatenate(features_test, axis=0)
print(features_test.shape)

os.makedirs("transfer/features/", exist_ok=True)
np.save("transfer/features/cifar100_resnext_train.npy", features_train)
np.save("transfer/features/cifar100_resnext_test.npy", features_test)
np.save("transfer/features/cifar100_resnext_mean.npy", mean_cifar100)
np.save("transfer/features/cifar100_resnext_var.npy", var_cifar100)

mean = np.mean(features_train, axis=0)
var = np.var(features_train, axis=0)

features_train_norm = (features_train - mean) / np.sqrt(var + 1e-5)
features_test_norm = (features_test - mean) / np.sqrt(var + 1e-5)

features_train_norm2 = (features_train - mean_cifar100) / np.sqrt(var_cifar100 + 1e-5)
features_test_norm2 = (features_test - mean_cifar100) / np.sqrt(var_cifar100 + 1e-5)

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train, ytrain)
    print(C, clf.score(features_train, ytrain), clf.score(features_test, ytest))

    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm, ytrain)
    print(C, clf.score(features_train_norm, ytrain), clf.score(features_test_norm, ytest))

    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm2, ytrain)
    print(C, clf.score(features_train_norm2, ytrain), clf.score(features_test_norm2, ytest))

