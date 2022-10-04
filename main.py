import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.den_model import *
from models.vgg_model import *
from models.res_model import *
from models.py_model import *

from pre_data import *
from args import build_parser

import sys

def lr_scheduler(optimizer, early, l):
    lr = l
    if early.counter > 6:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def train():
    
    parser = build_parser()
    args = parser.parse_args()
    config = args
    
    gpu = 'cuda:' + str(config.gpu)
    device = torch.device(gpu)
    
    if config.mode == "train":
        is_train = True
    else:
        is_train = False
    
    number_of_classes = 10
    train_mode = config.train_mode
    batch_size = config.batch_size
    epoch = config.epochs
    
    if config.dataset == "mnist":                                                                                                   # Load data from dataset
        train_loader, valid_loader, test_loader = Load_MNIST(train_mode, bs=batch_size)
    elif config.dataset == "cifar10":
        train_loader, valid_loader, test_loader = Load_Cifar10(train_mode, bs=batch_size)
    elif config.dataset == "cifar100":
        train_loader, valid_loader, test_loader = Load_Cifar100(train_mode, bs=batch_size)
        number_of_classes = 100
    else:
        train_loader, valid_loader, test_loader = Load_FMNIST(train_mode, bs=batch_size)
        
    if is_train:                                                                                                                    # Train phase
        early = EarlyStopping(patience=config.patience)
        
        train_model = config.train_model
        
        if train_model == 'vggnet':                                                                                                 # Prepare model
            if config.dataset == "mnist" or config.dataset == "fmnist":
                model = VGG("VGG16m", config.dataset, nc=number_of_classes)
            else:
                model = VGG("VGG16", config.dataset, nc=number_of_classes)
        elif train_model == 'resnet':
            model = ResNet50(config.dataset, nc=number_of_classes)
        elif train_model == 'densenet':
            model = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=number_of_classes, data=config.dataset)
        else:
            model = PyramidNet(dataset=config.dataset, depth=32, alpha=200, num_classes=number_of_classes, bottleneck=True)
        
        model = model.to(device)
        
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        loss_arr = []
        
        if train_mode < 3:
            train = train_loader

        for i in range(epoch):                                                                                                      # Train start
            model.train()
            print("=====", i, "Step of ", epoch, "=====")
            
            if train_mode == 3 or train_mode == 4:                                                                                  # Rebuild Batch data
                if config.dataset == "mnist":
                    train, _, _ = Load_MNIST(train_mode, bs=batch_size)
                elif config.dataset == "cifar10":
                    train, _, _ = Load_Cifar10(train_mode, bs=batch_size)
                elif config.dataset == "cifar100":
                    train, _, _ = Load_Cifar100(train_mode, bs=batch_size)
                else:
                    train, _, _ = Load_FMNIST(train_mode, bs=batch_size)          
            
            for j, batch in enumerate(train):
                x, y_ = batch[0].to(device), batch[1].to(device)
                #lr_scheduler(optimizer, early)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(output,y_)
                loss.backward()
                optimizer.step()

            if i % 10 ==0:
                loss_arr.append(loss.cpu().detach().numpy())

            correct = 0
            total = 0
            valid_loss = 0
            best_acc = 0
            
            model.eval()
            with torch.no_grad():                                                                                                   # Valid phase
                for image,label in valid_loader:
                    x = image.to(device)
                    y = label.to(device)
                        
                    output = model.forward(x)
                    valid_loss += loss_func(output, y)
                    _,output_index = torch.max(output,1)

                    total += label.size(0)
                    correct += (output_index == y).sum().float()
                print("loss : ", {valid_loss/total})
                train_acc = "Accuracy of Test Data: {}%".format(100*correct/total)
                print(train_acc)
                if correct/total > best_acc and config.save_model:
                    best_acc = correct/total
                    torch.save(model, './output/model')
                    print("model saved")

            early((valid_loss/total), model)
            
            if early.early_stop:
                print("stop")
                break
            scheduler.step()
    else:                                                                                                                           # Test phase
        model = torch.load('model').to(device)
        model.eval()
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
            c = (predict == batch[1]).squeeze()

                
        print(correct, total_cnt)
        valid_acc = correct / total_cnt
        print(f"\nTest Acc : { valid_acc }")    
        
if __name__ == '__main__':
    train()