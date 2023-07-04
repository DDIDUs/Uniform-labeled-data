from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
import random
import numpy as np

from sklearn.model_selection import train_test_split

def Prepare_data(train_mode, train_dataset, valid_dataset, bs, dataset, aug = None):
    
    if dataset != 'cifar100':
        label = [i for i in range(10)]
    elif dataset == 'Imagenet':
        label = [i for i in range(1000)]
    else:
        label = [i for i in range(100)]
    
    train_data_len = []
    train_data_by_label = []
    train_loader = []
    valid_data_by_label = []
    valid_len = []
    temp_aug = []
    
    if train_mode != 1:
        for i in label:
            train_data_by_label.append([])
            valid_data_by_label.append([])

        for i in train_dataset:
            train_data_by_label[i[1]].append(i)

        for i in valid_dataset:
            valid_data_by_label[i[1]].append(i)

        for i in label:
            random.shuffle(train_data_by_label[i])
            random.shuffle(valid_data_by_label[i])
            train_data_len.append(len(train_data_by_label[i]))
            valid_len.append(len(valid_data_by_label[i]))

        if aug != None:
            temp_aug.append([[] for _ in label])
            for i in aug:
                temp_aug[i[1]].append(i)
            for i in label:
                random.shuffle(temp_aug[i])
            for i in label:                                                                                                 # Aug append
                train_data_by_label[i] += random.sample(temp_aug[i], int(len(train_data_by_label[i])*0.2))
                valid_data_by_label[i] += random.sample(temp_aug[i], int(len(valid_data_by_label[i])*0.2))

    if train_mode == 1:                                                                                                 # 기존
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
    elif train_mode == 2 or train_mode == 3:                                                                            # 단순, 에폭 균등
        for i in train_data_by_label:
            train_loader += i
        train_loader = torch.utils.data.DataLoader(tuple(train_loader), batch_size=bs, shuffle=True, pin_memory = True, num_workers=4)
        
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, pin_memory = True, num_workers=4)
    return train_loader, valid_loader

dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'Imagenet' : torchvision.datasets.ImageFolder
}

def loadData(dataset_name = None, train_mode=None, batch_size=200, applyDataAug = False):

    transforms_Imagenet = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    aug = None
    
    if applyDataAug:                                                                                                  # 데이터 증강
        aug_data = torch.load("aug_{}.pt".format(dataset_name))
        aug = aug_data

    if dataset_name != "Imagenet" and train_mode == "train":
        dataset = dataset_load_func[dataset_name](root='./data/{}'.format(dataset_name), 
                                                  train=True, 
                                                  download=True,
                                                  transform=transforms.ToTensor())
        train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    elif dataset_name == "Imagenet" and train_mode == "train":
        train_dataset = dataset_load_func[dataset_name]('./data/{}/train'.format(dataset_name), 
                                                        transform=transforms_Imagenet)
        valid_dataset = dataset_load_func[dataset_name]('./data/{}/val'.format(dataset_name), 
                                                       transform=transforms_Imagenet)
    else:
        test_dataset = dataset_load_func[dataset_name](root='./data/{}'.format(dataset_name), 
                                                       train=False, 
                                                       download=True,
                                                       transform=transforms.ToTensor())

    if train_mode == "train":
        train_loader, valid_loader = Prepare_data(train_mode, train_dataset, valid_dataset, batch_size, dataset_name, aug)
        return train_loader, valid_loader
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return test_loader
