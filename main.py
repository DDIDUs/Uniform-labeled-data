import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from models.den_model import *
from models.vgg_model import *
from models.res_model import *
from models.py_model import *

from pre_data import *
from args import build_parser

import sys

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, dir='output'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dir = dir

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '{}/checkpoint_val_{:.2f}.pt'.format(self.dir,val_loss))
        torch.save(model, '{}/best.pt'.format(self.dir))
        self.val_loss_min = val_loss

def lr_scheduler(optimizer, early, l):
    lr = l
    if early.counter > 6:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def train(args, repeat_index):

    config = args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.mode == "train":
        is_train = True
    else:
        is_train = False
    
    if config.dataset == "cifar100":
        number_of_classes = 100
    else:
        number_of_classes = 10
    
    if config.Augmentation==True:
        output_dir = "./output/{}/{}/r{}_m{}_aug".format(config.dataset,config.train_model, repeat_index, config.train_mode)
    else:
        output_dir = "./output/{}/{}/r{}_m{}".format(config.dataset,config.train_model, repeat_index, config.train_mode)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, valid_loader, test_loader  = loadData(dataset_name=config.dataset, train_mode=config.train_mode, bs=config.batch_size, applyDataAug=config.Augmentation)
    ''' (추후 지워야 할 코드)
    if config.dataset == "mnist":                                                                                                   # Load data from dataset
        train_loader, valid_loader, test_loader = Load_MNIST(train_mode, bs=batch_size)
    elif config.dataset == "cifar10":
        if aug==False:
            train_loader, valid_loader, test_loader = Load_Cifar10(train_mode, bs=batch_size)
        if aug == True:
            train_loader, valid_loader, test_loader = Load_Cifar10_Aug(train_mode, bs=batch_size)
    elif config.dataset == "cifar100":
        if aug==False:
            train_loader, valid_loader, test_loader = Load_Cifar100(train_mode, bs=batch_size)
        if aug == True:
            train_loader, valid_loader, test_loader = Load_Cifar100_Aug(train_mode, bs=batch_size)
        number_of_classes = 100
    else:
        train_loader, valid_loader, test_loader = Load_FMNIST(train_mode, bs=batch_size)
    '''
    if is_train:                                                                                                                    # Train phase
        early = EarlyStopping(patience=config.patience, dir=output_dir)
        
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)
        loss_arr = []
        
        if config.train_mode < 3:
            train = train_loader

        best_acc = 0
        for i in range(config.epochs):                                                                                                      # Train start
            model.train()
            print("=====", i, "Step of ", config.epochs, "=====")
            
            if config.train_mode == 3 or config.train_mode == 4:
                train, _, _  = loadData(dataset_name=config.dataset,train_mode=config.train_mode,bs=config.batch_size,applyDataAug=config.Augmentation)# Rebuild Batch data
                ''' (추후 지워야 할 코드)
                if config.dataset == "mnist":
                    train, _, _ = Load_MNIST(train_mode, bs=batch_size)
                elif config.dataset == "cifar10":
                    if aug == False:
                        train, _, _ = Load_Cifar10(train_mode, bs=batch_size)
                    if aug == True:
                        train, _, _ = Load_Cifar10_Aug(train_mode, bs=batch_size)
                elif config.dataset == "cifar100":
                    if aug == False:
                        train, _, _ = Load_Cifar100(train_mode, bs=batch_size)
                    if aug == True:
                        train, _, _ = Load_Cifar100_Aug(train_mode, bs=batch_size)
                else:
                    train, _, _ = Load_FMNIST(train_mode, bs=batch_size)          
                '''
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
                train_acc = "Accuracy against Validation Data: {:.2f}%, Valid_loss: {:.2f}".format(100*correct/total, valid_loss)
                print(train_acc)

                current_acc = (correct / total) * 100
                if current_acc > best_acc:
                    print(" Accuracy increase from {:.2f}% to {:.2f}%. Model saved".format(best_acc, current_acc))
                    best_acc = current_acc
                    torch.save(model, './{}/epoch_{}_acc_{:.2f}_loss_{:.2f}.pt'.format(output_dir,i,best_acc,valid_loss))
            early(valid_loss, model)

            if early.early_stop:
                print("stop")
                break
            scheduler.step()
    else:
        mymodel = '{}/best.pt'.format(output_dir)
        # Test phase
        model = torch.load(mymodel).to(device)
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
        print("\nTest Acc : {}, {}".format(valid_acc,output_dir))
        
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    repeat_num = args.repeat_num
    for index in range(repeat_num):
        train(args=args, repeat_index=index)