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
