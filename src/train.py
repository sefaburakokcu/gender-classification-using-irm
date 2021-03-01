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
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from gender_dataset import GenderDataset

from mobilenetv2 import mobilenet_v2
        


def train(net, train_dataloader, epochs, weights_dir, resume='', filename="weights", checkpoint_frequency=None, val_dataloader=None):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr)
    
    if len(resume) != 0:
        checkpoint = torch.load(resume)
        optimizer_state_dict = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        best_validation_loss = checkpoint["best_validation_loss"]
        model_state_dict = checkpoint["state_dict"]
        
        net.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        start_epoch = 0
        best_validation_loss = np.inf
        
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                               milestones=[int(epochs/5), 2*int(epochs/5),
                                                           3*int(epochs/5), 4*int(epochs/5)])
    
    training_loss_list, validation_loss_list = [], []
    training_acc_list, validation_acc_list = [], []
    checkpoint_num = 0
    iteration = 0
    running_loss = 0
    
    if checkpoint_frequency is None:
        checkpoint_frequency = int(len(train_dataloader)/11)
    
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        running_acc = 0
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
            
            training_loss = running_loss/(i+1)
            
            running_acc += acc(outputs, labels)
            training_acc = running_acc/(i+1)
            
            # print(f'minibatch:{i}, epoch:{epoch}, iteration:{iteration}, \
            #       training_error:{training_loss:.4f}, training_acc:{training_acc}')
                                    
            if (iteration+1) % checkpoint_frequency == 0 and val_dataloader is not None:                                
                validation_loss, validation_acc = validate(net, val_dataloader)                
                
                print(f"batch:{i}, epoch:{epoch}, iteration:{iteration}, lr:{lr}, \
                      training_error:{training_loss:.4f}, validation_error:{validation_loss:.4f}, \
                      training_acc:{training_acc:.2f}, validation_acc:{validation_acc:.2f}")
                
                if validation_loss < best_validation_loss:
                    checkpoint = {'epoch':epoch,'state_dict':net.state_dict(),
                                  'optimizer':optimizer.state_dict(),
                                  'best_validation_loss':validation_loss,
                                  'state_dict':net.state_dict()}
                    torch.save(checkpoint, f'{weights_dir}{filename}_{epoch}_{checkpoint_num}.pth')
                    best_validation_loss = validation_loss 

                checkpoint_num += 1
            
            iteration += 1
            
        training_loss_list.append(round(training_loss,4))  
        training_acc_list.append(training_acc)
        
        if val_dataloader is not None:  
            validation_loss_list.append(round(validation_loss,4))
            validation_acc_list.append(validation_acc)
            
    checkpoint = {'epoch':epoch,'state_dict':net.state_dict(),
                                  'optimizer':optimizer.state_dict(),
                                  'best_validation_loss':best_validation_loss,
                                  'state_dict':net.state_dict()}    
    torch.save(checkpoint, f'{weights_dir}{filename}_final.pth')
    
    return net, training_loss_list, validation_loss_list, training_acc_list, validation_acc_list


def validate(net, dataloader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
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
    plt.figure()
    plt.plot(list(range(1,len(y1)+1)), y1, label=f'training_{ylabel}')
    plt.plot(list(range(1,len(y2)+1)), y2, label=f'validation_{ylabel}')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{weights_dir}{ylabel}.png")
    plt.show()


def visualize_samples(dataloader):
    single_batch = next(iter(dataloader))
    single_batch_grid = torchvision.utils.make_grid(single_batch[0], nrow=int(np.sqrt(single_batch[0].shape[0])))
    plt.figure(figsize = (10,10))
    plt.imsave(f"{weights_dir}samples.png", single_batch_grid.permute(1, 2, 0).numpy())
    plt.imshow(single_batch_grid.permute(1, 2, 0))
    

def test(net, dataloader):
    net.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)        
            y_test_pred = net(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1) 
            
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            
    y_pred_list = [j for i in y_pred_list for j in i]
    y_true_list = [j for i in y_true_list for j in i]
    net.train()
    return y_pred_list, y_true_list


def print_test_results(y_pred_list, y_true_list, index2classes):
    print(classification_report(y_true_list, y_pred_list))
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=index2classes, index=index2classes)
    fig, ax = plt.subplots(figsize=(7,5))         
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    plt.show()
    
    
def inference(net, dataloader):
    net.eval()
    single_batch = next(iter(dataloader))
    sample_num = single_batch[0].shape[0]
    cols = int(np.sqrt(sample_num))
    rows = cols
    img_num = cols*rows
    
    # predict
    with torch.no_grad():
        results = net(single_batch[0].to(device))
        preds = torch.log_softmax(results, dim = 1)
        _, preds = torch.max(preds, dim = 1) 
        
    net.train()
    for i in range(img_num):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(single_batch[0][i].permute(1, 2, 0).numpy())
        plt.title("{}".format("F" if preds[i]==0 else "M"))
        plt.axis("off")
    plt.savefig(f"{weights_dir}result.png")
    plt.show()


# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN,self).__init__()
# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(3,96,kernel_size=7,stride=4),
# 			nn.BatchNorm2d(96),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=3,stride=2))
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(96,256,kernel_size=5,padding=2),
# 			nn.BatchNorm2d(256),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=3,stride=2))
# 		self.layer3 = nn.Sequential(
# 			nn.Conv2d(256,384,kernel_size=3,padding=1),
# 			nn.BatchNorm2d(384),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=3,stride=2))
# 		self.fc1 = nn.Linear(384*6*6,512)
# 		self.fc2 = nn.Linear(512,512)
# 		self.fc3 = nn.Linear(512,2)

# 	def forward(self,x):
# 		out = self.layer1(x)
# 		out = self.layer2(out)
# 		out = self.layer3(out)
# 		out = out.view(out.size(0),-1)
# 		#print out.size()
# 		out = F.dropout(F.relu(self.fc1(out)))
# 		out = F.dropout(F.relu(self.fc2(out)))
# 		out = self.fc3(out)

# 		return out
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binary Gender Classification')
    parser.add_argument('--data_dir', type=str, default='/home/sefa/workspace/projects/datasets/gender_dataset_kaggle/')
    parser.add_argument('--weights_dir', type=str, default='./weights_noisylabels_mobv2(1.0)_erm/')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    weights_dir = args.weights_dir
    resume = args.resume
    
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = 0.0001  # initial learning rate
    
    model = mobilenet_v2(width_mult=1.0, pretrained=True)
    in_features = model.classifier[1].in_features
    classifier = nn.Sequential(nn.Linear(in_features, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                # nn.Linear(512, 512),
                                # nn.ReLU(),
                                # nn.Dropout(0.2),
                                nn.Linear(512, 2))
    model.classifier = classifier
    # model=CNN()
    # model.load_state_dict(torch.load("/home/sefa/workspace/projects/face_projects/gender-classification-using-irm/src/weights_orglabelsreal_mobv2(1.0)_erm_finalll/weights_3_33.pth")["state_dict"])
    model = model.to(device)
    print(model) 
    
    transform = transforms.Compose([
                                    transforms.Resize((112,112)),
                                    # transforms.CenterCrop(227),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])       

    trainset = GenderDataset(data_dir, train=True, transform=transform,
                              target_transform=None, convert=False, force=True,
                              noisy_labels = True)  
    # trainset = torchvision.datasets.ImageFolder(data_dir+"train/", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)    
        
        
    testset = GenderDataset(data_dir, train=False, transform=transform,
                            target_transform=None, convert=False, force=True,
                            noisy_labels = True)  
    # testset = torchvision.datasets.ImageFolder(data_dir+"test/", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=8) 
    
    visualize_samples(trainloader)
    
    net, training_loss_list, validation_loss_list, training_acc_list, \
    validation_acc_list = train(model, trainloader, num_epochs, weights_dir, resume,
                                checkpoint_frequency=None, val_dataloader=testloader)
       
    
    plot_results(training_loss_list, validation_loss_list)
    plot_results(training_acc_list, validation_acc_list, ylabel='accuracy')
    
    y_pred_list, y_true_list = test(model, testloader)
    
    index2classes = {0:'female', 1:'male'}
    print_test_results(y_pred_list, y_true_list, index2classes)

    inference(model, testloader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    