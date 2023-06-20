import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import argparse
from tqdm import tqdm

#--- Prepare CIFAR-10 dataset ---#

def prepare_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, num_workers=2)

    valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, 
                                            transform=transform_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                batch_size=512,
                                                shuffle=False)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, valid_dataloader, testloader

#--- Define our neural network ---#
class Our_Neural_Network(nn.Module):
    def __init__(self, in_channels, n_classes, dropout):
        super(Our_Neural_Network, self).__init__()
        #--- YOUR CODE HERE ---#
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #according to the intructions, we should drop the 4096 FC layer
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 1 * 1, n_classes)#the features of the last pooling are 1*1*512
        )

    def forward(self, x):
        #--- YOUR CODE HERE ---#
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.fc_layers(x)
        return x

def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for data in tqdm(trainloader, total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion):
    model.eval()
    # we need two lists to keep track of class-wise accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # calculate the accuracy for each class
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # print the accuracy for each class after evey epoch
    # the values should increase as the training goes on
    print('\n')
    for i in range(10):
        print(f"Accuracy of class {i}: {100*class_correct[i]/class_total[i]}")

    
    return epoch_loss, epoch_acc

def checkpoint(model, filename, optimizer):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)
#--- Create our neural network ---#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Our_Neural_Network(3, 10, dropout=0.2)
net = net.to(device)



#--- Define loss function and optimizer ---#
criterion = nn.CrossEntropyLoss()
loss = optim.SGD(net.parameters(),lr = 0.1, momentum=0.9, weight_decay=5e-4)
#--- YOUR CODE HERE ---#
if __name__ == "__main__":
    trainloader, valid_dataloader, testloader = prepare_data()
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_acc, best_loss = 0.0, 1000
    for epoch in range(0, 50):

        #--- TRAIN STEP ---#
        print('Starting training epoch: %d' % epoch)
        #--- YOUR CODE HERE ---#
        train_epoch_loss, train_epoch_acc = train(net, trainloader, 
                                                loss, criterion)
        print("Finished training epoch: %d" % epoch)

        #--- EVAL STEP ---#
        print('Starting evaluation at epoch: %d' % epoch)
        #--- YOUR CODE HERE ---#
        valid_epoch_loss, valid_epoch_acc = validate(net, valid_dataloader,  
                                                    criterion)
        #print("Validation accuracy at epoch %d: %.3f " % (epoch, 100.*correct/total))
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print('\n')
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
                    
        #--- Saving checkpoint ---#
        if valid_epoch_acc > best_acc: 
            best_acc = valid_epoch_acc
            print('Saving the checkpoint as the performance is better than previous best..')
            checkpoint(net, f"checkpoint/best_ckpt.pth", loss)
        #--- YOUR CODE HERE ---#



    #--- Saving plots with accuracy and losses curves ---#
    #--- YOUR CODE HERE ---#

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.jpg')
    plt.show()
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/losses.jpg')
    plt.show()
    
    print('TRAINING COMPLETE')
