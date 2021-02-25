"""
Created on Wed Feb 24 09:16:56 2021

@author: sefa
"""

import os
import copy
import glob
import argparse
import seaborn as sns
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from gender_dataset import GenderDataset

from mobilenetv2 import mobilenet_v2
        


def train(net, train_dataloader, epochs, weights_dir, filename="weights", checkpoint_frequency=None, val_dataloader=None):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    
    training_loss_list, validation_loss_list = [], []
    training_acc_list, validation_acc_list = [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0
    best_validation_loss = np.inf
    
    if checkpoint_frequency is None:
        checkpoint_frequency = int(len(train_dataloader)/11)
    
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
        
    
    for epoch in range(epochs):
        running_loss = 0
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
            
            training_loss = running_loss/(iteration+1)
            training_acc = acc(outputs, labels)
            
            # print(f'minibatch:{i}, epoch:{epoch}, iteration:{iteration}, \
            #       training_error:{training_loss:.4f}, training_acc:{training_acc}')
                                    
            if (iteration+1) % checkpoint_frequency == 0 and val_dataloader is not None:                                
                validation_loss, validation_acc = validate(net, val_dataloader)                
                
                print(f'minibatch:{i}, epoch:{epoch}, iteration:{iteration}, \
                      training_error:{training_loss:.4f}, validation_error:{validation_loss:.4f}, \
                      training_acc:{training_acc:.2f}, validation_acc:{validation_acc:.2f}')
                
                if validation_loss < best_validation_loss:
                    torch.save(net.state_dict(), f'{weights_dir}{filename}_{epoch}_{checkpoint}.pt')
                    best_validation_loss = validation_loss 

                checkpoint += 1
                running_loss = 0
            
            iteration += 1
            
        training_loss_list.append(round(training_loss,4))  
        training_acc_list.append(training_acc)
        
        if val_dataloader is not None:  
            validation_loss_list.append(round(validation_loss,4))
            validation_acc_list.append(validation_acc)
        
    torch.save(net.state_dict(), f'{weights_dir}{filename}_final.pt')
    
    return net, training_loss_list, validation_loss_list, training_acc_list, validation_acc_list


def validate(net, dataloader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
#            labels = labels.unsqueeze(1).float()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            validation_acc = acc(outputs, labels)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            total_acc += validation_acc
    
    net.train()
    return total_loss/(i+1), total_acc/(i+1)


def acc(preds, targets):
    preds_tag = torch.log_softmax(preds, dim = 1)
    _, preds_tags = torch.max(preds_tag, dim = 1) 
    correct_results_sum = (preds_tags == targets).sum().float()    
    acc = correct_results_sum/targets.shape[0]
    acc = torch.round(acc * 100)
    return acc


def plot_results(y1, y2, xlabel='Epochs', ylabel='loss'):
    plt.plot(list(range(1,len(y1)+1)), y1, label=f'training_{ylabel}')
    plt.plot(list(range(1,len(y2)+1)), y2, label=f'validation_{ylabel}')
    plt.legend(ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
# https://towardsdatascience.com/pytorch-vision-binary-image-classification-d9a227705cf9


def visualize_samples(dataloader):
    single_batch = next(iter(dataloader))
    single_batch_grid = torchvision.utils.make_grid(single_batch[0], nrow=int(np.sqrt(single_batch[0].shape[0])))
    plt.figure(figsize = (10,10))
    plt.imshow(single_batch_grid.permute(1, 2, 0))
    

def test(model, dataloader):
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)        
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1) 
            
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            
    y_pred_list = [j for i in y_pred_list for j in i]
    y_true_list = [j for i in y_true_list for j in i]
    
    return y_pred_list, y_true_list


def print_test_results(y_pred_list, y_true_list, index2classes):
    print(classification_report(y_true_list, y_pred_list))
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=index2classes, index=index2classes)
    fig, ax = plt.subplots(figsize=(7,5))         
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binary Gender Classification')
    parser.add_argument('--data_dir', type=str, default='/home/sefa/Downloads/archive/')
    parser.add_argument('--weights_dir', type=str, default='./weights_original_erm_0.25/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    weights_dir = args.weights_dir
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = 0.0001  # initial learning rate
    
    model = mobilenet_v2(width_mult=0.25, pretrained=False)
    classifier = model.classifier
    model.classifier[1] = torch.nn.Linear(classifier[1].in_features, 2)    
    model = model.to(device)
    print(model) 
    
    transform = transforms.Compose([
     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])       

    trainset = GenderDataset(data_dir, train=True, transform=transform,
                             target_transform=None, convert=False, force=True)    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)    
        
        
    testset = GenderDataset(data_dir, train=False, transform=transforms.ToTensor(),
                            target_transform=None, convert=False, force=True)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=8) 
    
    visualize_samples(trainloader)
    
    net, training_loss_list, validation_loss_list, training_acc_list, validation_acc_list = train(model, trainloader, num_epochs, weights_dir, checkpoint_frequency=None, val_dataloader=testloader)
       
    
    plot_results(training_loss_list, validation_loss_list)
    plot_results(training_acc_list, validation_acc_list, ylabel='accuracy')
    
    y_pred_list, y_true_list = test(model, testloader)
    
    index2classes = {0:'female', 1:'male'}
    print_test_results(y_pred_list, y_true_list, index2classes)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    