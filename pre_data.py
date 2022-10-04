from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
import random
import numpy as np

from sklearn.model_selection import train_test_split

def Prepare_data(train_mode, train_dataset, valid_dataset, bs, dataset):
    
    if dataset != 'cifar100':
        label = [i for i in range(10)]
    else:
        label = [i for i in range(100)]
    
    train_data_len = []
    train_label = []
    train_loader = []
    min_len = 0
    val_min_len = 0
    valid_label = []
    valid_len = []
    
    for i in label:
        train_label.append([])
        valid_label.append([])
    
    if train_mode != 1:
        for i in train_dataset:
            train_label[i[1]].append(i)

        for i in valid_dataset:
            valid_label[i[1]].append(i)

        for i in label:
            random.shuffle(train_label[i])
            random.shuffle(valid_label[i])
            train_data_len.append(len(train_label[i]))
            valid_len.append(len(valid_label[i]))
        min_len = min(train_data_len)

        val_min_len = min(valid_len)
        val_temp = []

        for i in range(val_min_len):
            w = []
            for t in valid_label:
                w.append(t.pop())
            val_temp+=w
        valid_dataset = val_temp

    if train_mode == 1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    elif train_mode == 2:
        for i in train_label:
            t = i[:min_len]
            train_loader += t
        train_loader = torch.utils.data.DataLoader(tuple(train_loader), batch_size=bs, shuffle=True, num_workers=8)
    elif train_mode == 4:
        #train_loader = torch.utils.data.DataLoader(tuple(sorted(train_dataset, key=lambda x: x[1])), batch_size=256, shuffle=False, num_workers=8)
        temp = []
        for i in range(min_len):
            w = []
            for t in train_label:
                w.append(t.pop())
            temp+=w
        train_loader = torch.utils.data.DataLoader(tuple(temp), batch_size=bs, shuffle=False, num_workers=8)
    else:        
        for i in train_label:
            t = i[:min_len]
            train_loader += t
            #train_loader.append(torch.utils.data.DataLoader(tuple(t), batch_size=bs, shuffle=True, num_workers=8))
        train_loader = torch.utils.data.DataLoader(tuple(train_loader), batch_size=bs, shuffle=True, num_workers=8)
        
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader

def Load_Cifar10(train_mode=None, bs = 200):
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, shuffle=False)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

    train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, bs, 'cifar10')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader, test_loader

def Load_MNIST(train_mode = None, bs = 200):
    
    dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, bs, 'mnist')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader, test_loader


def Load_FMNIST(train_mode = None, bs = 200):
    
    dataset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    testset = torchvision.datasets.FashionMNIST(root='./data/fmnist', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, bs, 'fmnist')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader, test_loader

def Load_Cifar100(train_mode = None, bs = 200):
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.25, shuffle=False)
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
    
    train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, bs, 'cifar100')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader, test_loader