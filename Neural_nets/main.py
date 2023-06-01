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


#--- Prepare CIFAR-10 dataset ---#
print('==> Preparing data..')
transform_train = transforms.Compose([
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

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


#--- Define our neural network ---#
class Our_Neural_Network(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Our_Neural_Network, self).__init__()
        #--- YOUR CODE HERE ---#
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #according to the intructions, we should drop the 4096 FC layer
        self.fc_layers = nn.Linear(512 * 1 * 1, n_classes) #the features of the last pooling are 1*1*512

    def forward(self, x):
        #--- YOUR CODE HERE ---#
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) #flatten
        x = self.fc_layers(x)
        return x

#--- Create our neural network ---#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Our_Neural_Network()
net = net.to(device)



#--- Define loss function and optimizer ---#
criterion = nn.CrossEntropyLoss()
loss = optim.SGD(lr = 0.1, momentum=0.9, weight_decay=5e-4)
#--- YOUR CODE HERE ---#

for epoch in range(0, 50):

    #--- TRAIN STEP ---#
    print('Starting training epoch: %d' % epoch)
    #--- YOUR CODE HERE ---#
    print("Finished training epoch: %d" % epoch)

    #--- EVAL STEP ---#
    print('Starting evaluation at epoch: %d' % epoch)
    #--- YOUR CODE HERE ---#
    #print("Validation accuracy at epoch %d: %.3f " % (epoch, 100.*correct/total))
     
                   
    #--- Saving checkpoint ---#

    print('Saving the checkpoint as the performance is better than previous best..')
    #--- YOUR CODE HERE ---#



#--- Saving plots with accuracy and losses curves ---#
#--- YOUR CODE HERE ---#
