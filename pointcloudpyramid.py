

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2, PIL
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Pyramid_Layer_1(nn.Module):
  def __init__(self):
    super().__init__()

    self.lin1 = nn.Linear(in_features=1024,out_features=256*512)
    self.conv1D = nn.Conv1d(in_channels=512,out_channels=12, kernel_size = 1)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = x.reshape([-1,512, 256])
    x = F.relu(self.conv1D(x))
    x = torch.reshape(x, (-1, 256, 4, 3))

    return x

# summary(Pyramid_Layer_1().to(device), input_size = (1, 1024) )

class Pyramid_Layer_2(nn.Module):
  def __init__(self):
    super().__init__()

    self.lin1 = nn.Linear(in_features=512,out_features=128*128)
    self.conv1D = nn.Conv1d(in_channels=128,out_channels=6, kernel_size = 1)


  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = x.reshape([-1,128, 128])
    x = F.relu(self.conv1D(x))
    x = torch.reshape(x, (-1, 128, 2, 3))

    return x

# summary(Pyramid_Layer_2().to(device), input_size = (1, 512) )

class Pyramid_Layer_3(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(in_features=256,out_features=128*3)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = torch.reshape(x, (-1, 128, 1, 3))

    return x

# summary(Pyramid_Layer_3().to(device), input_size = (1, 256) )

class PointCloudPyramid(nn.Module):
  def __init__(self, layer1, layer2, layer3):
    super().__init__()

    self.input1 = nn.Linear(in_features=2000,out_features=1024)

    self.downSample1 = nn.Linear(in_features=1024,out_features=512)
    self.downSample2 = nn.Linear(in_features=512,out_features=256)

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3

  def forward(self, x):

    x1 = F.relu(self.input1(x))
    x2 = F.relu(self.downSample1(x1))
    x3 = F.relu(self.downSample2(x2))
    
    out1 = self.layer1(x1)
    out2 = self.layer2(x2)
    out3 = self.layer3(x3)

    y2 = out3 + out2
    y2 = y2.reshape((-1,256,1,3))
    y1 = out1 + y2
    y1 = y1.reshape((-1,256,4,3))
    outputPointCloud = y1.reshape((-1,1024,3))

    return outputPointCloud

# summary(PointCloudPyramid(Pyramid_Layer_1(), Pyramid_Layer_2(), Pyramid_Layer_3()).to(device), input_size = (1, 2000) )

