from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
import random
import numpy as np
import os.path as path
import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from sklearn.model_selection import train_test_split

def process_data(img_info, augmentation=False, mode = 1):
    data_len = []
    new_info = []
    label_len = [i for i in range(1000)]
    temp_aug = [[] for _ in label_len]
    if augmentation:
        for i in label_len:                                                                                                 # Aug append
            temp_aug[i] = random.sample(img_info[i], int(len(img_info[i])*0.2))        
        for i in temp_aug:
            for j in i:
                j[-1] = True
        for i in label_len:
            img_info[i] += temp_aug[i]

    for i in label_len:
        data_len.append(len(img_info[i]))
    for i in img_info:
        new_info += i
    random.shuffle(new_info)
    return new_info

class ImageData(Dataset):
    def __init__(self, data, root_dir, transform=None, aug_transform = None, augmentation=False):
        self.img_info = data
        self.root_dir = root_dir
        self.transform = transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        image = Image.open(self.img_info[idx][0]).convert("RGB")
        label = torch.tensor(self.img_info[idx][1])
        if self.img_info[idx][2]:
            image = self.aug_transform(image)
        else:
            image = self.transform(image)

        return image, label

dataset_load_func = {
    'mnist': torchvision.datasets.MNIST,
    'fmnist':torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'imagenet' : torchvision.datasets.ImageFolder
}

def pre_path(r_path, dataset, mode):
    label = []
    info = []
    tmp2 = []
    for (path, dir, files) in os.walk(r_path):
        if path == r_path:
            label = dir
            continue
        info = []
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            tmp = path.split("/")[-1]
            if ext == '.JPEG':
                info.append(["%s/%s" % (path, filename), label.index(tmp), False])
        tmp2.append(info)

    with open("data/{}/{}_path.pkl".format(dataset, mode),"wb") as f:
        pickle.dump(tmp2, f)
    
    return tmp2

def loadData(dataset_name = None, train_mode=None, batch_size=200, applyDataAug = False):

    transforms_Imagenet = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    transforms_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2))
            ]),
        transforms.ToTensor()
        ])

    if not path.exists("data/{}/train_path.pkl".format(dataset_name)) or not path.exists("data/{}/valid_path.pkl".format(dataset_name)):
        t_data = pre_path("data/{}/train".format(dataset_name), dataset_name, "train")
        v_data = pre_path("data/{}/val".format(dataset_name), dataset_name, "valid")
    else:
        with open("./data/{}/train_path.pkl".format(dataset_name), "rb") as f:
            t_data = pickle.load(f)
        with open("./data/{}/valid_path.pkl".format(dataset_name), "rb") as f:
            v_data = pickle.load(f)
            
    if train_mode == "train":
        train_data = ImageData(process_data(t_data, applyDataAug, train_mode), "data/imagenet", transforms_Imagenet, transforms_train, applyDataAug)
        valid_data = ImageData(process_data(v_data, applyDataAug, train_mode), "data/imagenet", transforms_Imagenet, transforms_train, applyDataAug)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)      
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, valid_loader
    else:
        if not path.exists("data/{}/test_path.pkl".format(dataset_name)):
            t_data = pre_path("data/{}/test".format(dataset_name), dataset_name, "test")
        test_data = ImageData(process_data(t_data, applyDataAug, train_mode), "data/imagenet", transforms_Imagenet, transforms_train, applyDataAug)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
        return test_loader