import sys
sys.path.insert(0, 'RepVGG')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as tf
import torch.optim as optim
from torchsummary import summary

from RepVGG.repvgg import create_RepVGG_A0
from RepVGG import se_block

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

MainBranch = create_RepVGG_A0().to(device)
weightsA0 = torch.load("Pretrained_Networks/RepVGG-A0-train.pth")
MainBranch.load_state_dict(weightsA0)

# print(model)
summary(MainBranch, (3, 128,128))


class AuxilaryBranchCNN(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.AvgPool2d(2, stride = 2)
    
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.AvgPool2d(2, stride = 2)

    self.lin1 = nn.Linear(in_features=32*32*3,out_features=1000)

  def forward(self, x):
    x = self.pool1(functional.relu(self.conv1(x)))
    x = self.pool2(functional.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = functional.relu(self.lin1(x))

    return x


AuxilaryBranch = AuxilaryBranchCNN().to(device)
print(AuxilaryBranch)
summary(AuxilaryBranch, (1, 128,128))

class PointCloudPyramid(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.layer1_lin1 = nn.Linear(in_features=2000,out_features=1024)
    self.layer1_lin2 = nn.Linear(in_features=1024,out_features=256*512)
    # self.layer1_conv1 = nn.Conv1d(in_channels=1024,out_channels=256*512, kernel_size=)


  def forward(self, x):

    # x = functional.relu(self.lin1(x))


    return x


PCP_Model = PointCloudPyramid().to(device)
print(PCP_Model)
summary(PCP_Model, (1, 2000))