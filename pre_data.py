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
    train_data_by_label = []
    train_loader = []
    min_len = 0
    val_min_len = 0
    valid_data_by_label = []
    valid_len = []
    
    for i in label:
        train_data_by_label.append([])
        valid_data_by_label.append([])
    
    if train_mode != 1:
        for i in train_dataset:
            train_data_by_label[i[1]].append(i)

        for i in valid_dataset:
            valid_data_by_label[i[1]].append(i)

        for i in label:
            random.shuffle(train_data_by_label[i])
            random.shuffle(valid_data_by_label[i])
            train_data_len.append(len(train_data_by_label[i]))
            valid_len.append(len(valid_data_by_label[i]))
        min_len = max(train_data_len)
        val_min_len = max(valid_len)

        for i in label:                                                                                                 # 업 
            train_data_by_label[i] += random.sample(train_data_by_label[i], min_len - len(train_data_by_label[i]))
            valid_data_by_label[i] += random.sample(valid_data_by_label[i], val_min_len - len(valid_data_by_label[i]))
        val_temp = []

        for i in range(val_min_len):
            w = []
            for t in valid_data_by_label:
                w.append(t.pop())
            val_temp+=w
        valid_dataset = val_temp

    if train_mode == 1:                                                                                                 # 기존
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    elif train_mode == 2 or train_mode == 3:                                                                            # 단순, 에폭 균등
        for i in train_data_by_label:
            t = i[:min_len]
            train_loader += t
        train_loader = torch.utils.data.DataLoader(tuple(train_loader), batch_size=bs, shuffle=True, num_workers=8)
    else:                                                                                                               # 배치 균등
        temp = []
        for i in range(min_len):
            w = []
            for t in train_data_by_label:
                w.append(t.pop())
            temp+=w
        train_loader = torch.utils.data.DataLoader(tuple(temp), batch_size=bs, shuffle=False, num_workers=8)
        
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=8)
    
    return train_loader, valid_loader

dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}

def loadData(dataset_name = None, train_mode=None, batch_size=200, applyDataAug = False):

    dataset = dataset_load_func[dataset_name](root='./data/{}'.format(dataset_name), 
                                              train=True, 
                                              download=True,
                                              transform=transforms.ToTensor())
    
    test_dataset = dataset_load_func[dataset_name](root='./data/{}'.format(dataset_name), 
                                                   train=False, 
                                                   download=True,
                                                   transform=transforms.ToTensor())

    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)

    if applyDataAug:                                                                                                  # 데이터 증강
        aug_data = torch.load("aug_{}.pt".format(dataset_name))
        aug, _ = train_test_split(aug_data, test_size=0.2, shuffle=False)
        train_dataset = train_dataset + aug

    train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, batch_size, dataset_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, valid_loader, test_loader
