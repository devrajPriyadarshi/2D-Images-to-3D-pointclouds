import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import auxilarynet as aux

#setup logging
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)

def validator(testloader,net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)[1]
            # the class with the highest energy is what we choose as prediction
            # perform max along dimension 1, since dimension 0 is batch dimension
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the 10000 test images: {100 * correct // total} %\n')
    return correct/total

def training(start_epoch , end_epoch , net , optimizer , criterion , testloader, trainloader):
    best_accuracy = -1.0
    for epoch in range(start_epoch, end_epoch):  

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)[1] # NOTICE: Aux Net outputs multiple values
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        current_accuracy = validator(testloader=testloader,net=net)
        if current_accuracy>best_accuracy:
            best_accuracy = current_accuracy
            
            torch.save(
                {'epoch':epoch, 
                'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                
                './Pretrained_Networks/Auxiliary_Network.pth')

    logging.info('Finished Training\n')
    logging.info(f"Saved the best network in \"./Pretrained_Networks\" Folder\n")


# start Traning!
if __name__ == "__main__":

    print("\n")
    print("------- Training Auxiliary Net -------")
    print("\n")

    # Setup Hardware:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != "cuda:0":
        logging.info(f"We are training on {device}\n")

    # Setup Training Parameters:
    dataset_path = "./DATASET/cifar-10-python"
    transform = tf.Compose( [tf.Grayscale(),
                             tf.Resize((128,128)),
                             tf.ToTensor(),
                             tf.Normalize((0.5), (0.5))])
    batch_size = 32
    start_epoch = 0
    end_epoch = 20
    lr = 0.001
    # momentum = 0.9

    logging.info(f"Loading CIFAR-10 dataset...\n")
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get Network:
    logging.info(f"Loading Auxiliary Branch...\n")
    net = aux.AuxilaryBranchCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Start Training:
    logging.info(f"Starting Training...")
    logging.info(f"Start Epoch = {start_epoch}, End Epoch = {end_epoch}:\n")
    training(start_epoch = start_epoch, end_epoch = end_epoch, net = net, optimizer = optimizer, criterion = criterion, testloader = testloader, trainloader = trainloader)