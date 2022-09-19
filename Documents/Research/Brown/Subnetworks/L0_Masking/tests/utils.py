from ..L0_Linear import L0UnstructuredLinear
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class HiLoDataset(Dataset):
    def __init__(self):
        high_samples = (torch.rand(1000, 10)/2) + .5
        high_class = torch.zeros(1000, dtype=torch.long).fill_(1)
        low_samples = torch.rand(1000, 10) / 2
        low_class = torch.zeros(1000, dtype=torch.long).fill_(0)
        self.samples = torch.cat((high_samples, low_samples))
        self.labels = torch.cat((high_class, low_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

class L0MLP(nn.Module):
    def __init__(self):
        super(L0MLP, self).__init__()
        self.model = nn.Sequential(
            L0UnstructuredLinear(10, 20),
            nn.ReLU(),
            L0UnstructuredLinear(20, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

    def train(self, train_bool):
        for layer in self.modules():
            try:
                layer.train(train_bool)
            except:
                continue

#Linear network to prune after training
class MLP(nn.Module):
    def __init__(self):
        super(L0MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

    def train(self, train_bool):
        for layer in self.modules():
            try:
                layer.train(train_bool)
            except:
                continue



class L0_Loss():
    def __init__(self, lamb=1.):
        self.error_loss = nn.CrossEntropyLoss()
        self.lamb = lamb

    def _get_model_l0(self, model):
        l0_loss = 0.0
        for layer in model.modules():
            if isinstance(layer, L0UnstructuredLinear):
                l0_loss += layer.num_parameters()
        return l0_loss

    def _get_model_prunable_params(self, model):
        max_L0 = 0.0
        for layer in model.modules():
            if isinstance(layer, L0UnstructuredLinear):
                max_L0 += layer.num_prunable_parameters()
        return max_L0

    def __call__(self, input, target, model):
        l_err = self.error_loss(input, target)
        l_complexity = self._get_model_l0(model)
        return l_err + self.lamb * l_complexity


def train_L0_Linear(lambda_numerator=1.):
    model = L0MLP()
    train_set = HiLoDataset()
    test_set = HiLoDataset()
    trainloader = DataLoader(train_set, shuffle=True, batch_size=2)
    testloader = DataLoader(test_set, batch_size=2)
    criterion = L0_Loss(lambda_numerator/len(train_set))
    optimizer = optim.Adam(model.parameters())

    for _ in range(3):
        for data in trainloader:
            ipts, labels = data
            optimizer.zero_grad()
            outputs = model(ipts)
            loss = criterion(outputs, labels, model)
            loss.backward()
            optimizer.step()

    total = 0.
    correct = 0.
    model.train(False)
    with torch.no_grad():
        for data in testloader:
            ipts, labels = data
            outputs = model(ipts)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct/total
    print(f"Accuracy: {acc}")
    return model, acc


def train_L0_probe():
    model = MLP()
    train_set = HiLoDataset()
    test_set = HiLoDataset()
    trainloader = DataLoader(train_set, shuffle=True, batch_size=2)
    testloader = DataLoader(test_set, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for _ in range(3):
        for data in trainloader:
            ipts, labels = data
            optimizer.zero_grad()
            outputs = model(ipts)
            loss = criterion(outputs, labels, model)
            loss.backward()
            optimizer.step()

    total = 0.
    correct = 0.
    model.train(False)
    with torch.no_grad():
        for data in testloader:
            ipts, labels = data
            outputs = model(ipts)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    unpruned_acc = correct/total
    print(f"Unpruned Accuracy: {unpruned_acc}")



    return model, acc
