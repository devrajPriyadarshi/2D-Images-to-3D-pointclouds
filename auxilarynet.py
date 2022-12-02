import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2, PIL
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AuxilaryBranchCNN(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.AvgPool2d(2, stride = 2)
    
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.AvgPool2d(2, stride = 2)

    self.flat = nn.Flatten()
    self.lin1 = nn.Linear(in_features=32*32*3,out_features=1000)


  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.flat(x)
    x = self.lin1(x)

    return x

class Ensemble(nn.Module):
  def __init__(self, Auxillary):
    super().__init__()
    
    self.Auxillary = Auxillary
    self.lin2 = nn.Linear(in_features=1000,out_features=10)

  def forward(self, x):
    x = F.relu(self.Auxillary(x))
    x = self.lin2(x)
    x = F.log_softmax(x, dim = 1)
    return x
