# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:16:56 2021

@author: sefa
"""

import copy
import glob
import argparse
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from PIL import Image
from gender_dataset import GenderDataset

from mobilenetv2 import mobilenet_v2
        


def train(net, train_dataloader, epochs, filename, checkpoint_frequency=10, val_dataloader=None):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    
    training_loss, validation_loss = [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            scheduler.step()
            optimizer.zero_grad()
#            labels = labels.unsqueeze(1).float()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            loss.backward()
            optimizer.step()
                                    
            if (iteration+1) % checkpoint_frequency == 0 and val_dataloader is not None:
                training_loss.append(running_loss/checkpoint_frequency)
                validation_loss.append(validate(net, val_dataloader))
                print(f'minibatch:{i}, epoch:{epoch}, iteration:{iteration}, training_error:{training_loss[-1]}, validation_error:{validation_loss[-1]}')
                torch.save(net.state_dict(), f'{filename}_{checkpoint}.pt')

                checkpoint += 1
                running_loss = 0
            
            iteration += 1

    return net, training_loss, validation_loss


def validate(net, dataloader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
#            labels = labels.unsqueeze(1).float()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
    
    net.train()
    return total_loss/(i+1)


def acc(preds, targets):
    preds_tag = torch.log_softmax(preds, dim = 1)
    _, preds_tags = torch.max(preds_tag, dim = 1) 
    correct_results_sum = (preds_tags == targets).sum().float()    
    acc = correct_results_sum/targets.shape[0]
    acc = torch.round(acc * 100)
    return acc


# https://towardsdatascience.com/pytorch-vision-binary-image-classification-d9a227705cf9

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binary Gender Classification')
    parser.add_argument('--data_dir', type=str, default='/home/sefa/Downloads/archive/')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 32
    num_epochs = 60
    lr = 0.0001  # initial learning rate
    
    model = mobilenet_v2(pretrained=True)
    classifier = model.classifier
    model.classifier[1] = torch.nn.Linear(classifier[1].in_features, 2)    
    model = model.to(device)
    print(model) 
    
    transform = transforms.Compose([
     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])       

    trainset = GenderDataset(data_dir, train=True, transform=transform,
                                     target_transform=None)    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)    
        
        
    testset = GenderDataset(data_dir, train=False, transform=transforms.ToTensor(),target_transform=transforms.ToTensor())   
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8) 

    train(model, trainloader, num_epochs, filename="gender",val_dataloader=testloader)
    