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

from auxilarynet import AuxilaryBranchCNN, Ensemble

from pointcloudpyramid import PointCloudPyramid, Pyramid_Layer_1, Pyramid_Layer_2, Pyramid_Layer_3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)


MainBranch = create_RepVGG_A0().to(device)
weightsA0 = torch.load("Pretrained_Networks/RepVGG-A0-train.pth")
MainBranch.load_state_dict(weightsA0)

AuxiliaryBranch = AuxilaryBranchCNN().to(device)
weights = torch.load("Pretrained_Networks/auxiliaryBranch-01.pth")
AuxiliaryBranch.load_state_dict(weights)

PCP = PointCloudPyramid(Pyramid_Layer_1(), Pyramid_Layer_2(), Pyramid_Layer_3()).to(device)
# weights = torch.load("Pretrained_Networks/auxiliaryBranch-01.pth")
# AuxiliaryBranch.load_state_dict(weights)

print("\n\n---------------- Main Branch ----------------")
summary(MainBranch, (3, 128,128))

print("\n\n---------------- Auxiliary Branch ----------------")
summary(AuxiliaryBranch, (1, 128,128))

print("\n\n---------------- Point Cloud Pyramid ----------------")
summary(PCP, (1, 2000))