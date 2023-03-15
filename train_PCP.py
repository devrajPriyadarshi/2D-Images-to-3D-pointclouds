import sys
sys.path.insert(0, 'RepVGG')
sys.path.insert(0, 'utils')

import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from RepVGG.repvgg import create_RepVGG_A0
from RepVGG import se_block

from auxilarynet import AuxilaryBranchCNN
from main_model import pointCloudGenerator
from pointcloudpyramid import PointCloudPyramid, Pyramid_Layer_1, Pyramid_Layer_2, Pyramid_Layer_3
from chamferdist import ChamferDistance

from validator import validator
from data_loaders import parseTrainData, parseValData, DatasetLoader
from custom_losses import TotalLoss
from CannyEdge import CannyEdgeDetection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

MainBranch = create_RepVGG_A0().to(device)
weightsA0 = torch.load("Pretrained_Networks/RepVGG-A0-train.pth")
MainBranch.load_state_dict(weightsA0)

AuxiliaryBranch = AuxilaryBranchCNN().to(device)
weightsAux = torch.load("Pretrained_Networks/Auxiliary_Network.pth")
AuxiliaryBranch.load_state_dict(weightsAux["model_state_dict"])

PCP = PointCloudPyramid(Pyramid_Layer_1(), Pyramid_Layer_2(), Pyramid_Layer_3()).to(device)

#setup logging
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)

def training(start_epoch , end_epoch , net , optimizer , criterion , testloader , trainloader):
    best_accuracy = -1.0
    for epoch in range(start_epoch, end_epoch):  

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            rgb_img, gt_pc = data

            rgb_img = rgb_img.to(device)
            gt_pc = gt_pc.to(device)
            edge_img = CannyEdgeDetection(rgb_img)

            optimizer.zero_grad()

            output = net(rgb_img, edge_img)
            loss = criterion(output, gt_pc)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 2000:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        current_accuracy = validator(testloader=testloader,net=net)
        if current_accuracy>best_accuracy:
            best_accuracy = current_accuracy
            
            torch.save(
                {'epoch':epoch, 
                'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                
                './Pretrained_Networks/PCP.pth')

    logging.info('Finished Training\n')
    logging.info(f"Saved the best network in \"./Pretrained_Networks\" Folder\n")


def testingLoaders(net, optimizer, criterion, testloader, trainloader):
    
    dataiter = iter(trainloader)
    rgb_img, edge_img, gt_pc = next(dataiter)

    rgb_img = rgb_img.to(device)
    edge_img = edge_img.to(device)
    gt_pc = gt_pc.to(device)

    optimizer.zero_grad()
    output = net(rgb_img, edge_img) 
    loss = criterion(output, gt_pc)
    
    print(loss)


# start Traning!
if __name__ == "__main__":

    print("\n")
    print("------- Training PCP -------")
    print("\n")

    # Setup Hardware:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != "cuda:0":
        logging.info(f"We are training on {device}\n")

    # Setup Training Parameters:
    transform = tf.Compose( [tf.Resize((128,128)),
                             tf.ToTensor()
                            #  tf.Normalize((0.5), (0.5))
                             ])
    batch_size = 24
    start_epoch = 0
    end_epoch = 20
    lr = 0.001

    logging.info(f"Loading Train Dataset dataset...\n")
    img_,mod_,ang_ = parseTrainData()
    trainset = DatasetLoader(model_paths = mod_, image_paths = img_, angel_paths = ang_, sourceTransform = transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    img_,mod_,ang_ = parseValData()
    testset = DatasetLoader(model_paths = mod_, image_paths = img_, angel_paths = ang_, sourceTransform = transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get Network:
    logging.info(f"Loading Point Cloud Pyramid...\n")
    net = pointCloudGenerator(MainBranch, AuxiliaryBranch, PCP)


    ## Parameter Freezing
    logging.info(f"Freezing All Parameters...")
    logging.info(f"And Un-Freezing:\n")

    parameterToTrain = []

    for x in net.parameters():
        x.requires_grad = False
    
    for x in net.pcp.parameters():
        x.requires_grad = True

    for name, x in net.named_parameters():
        if x.requires_grad == True:
            print(name)
            parameterToTrain.append(x)

    print("\n")
    logging.info(f"Starting Training...")
    logging.info(f"Start Epoch = {start_epoch}, End Epoch = {end_epoch}:\n")
    optimizer = optim.Adam(parameterToTrain, lr = lr)
    criterion = TotalLoss()

    testingLoaders(net = net, optimizer = optimizer, criterion = criterion, testloader = testloader, trainloader= trainloader)
    # training(start_epoch = start_epoch, end_epoch = end_epoch, net = net, optimizer = optimizer, criterion = criterion, testloader = testloader, trainloader= trainloader)