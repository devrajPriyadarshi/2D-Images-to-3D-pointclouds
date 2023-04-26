import sys
sys.path.insert(0, 'RepVGG')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as tf
import torch.optim as optim
from torchinfo import summary

from RepVGG.repvgg import create_RepVGG_A0
from RepVGG import se_block

from auxilarynet import AuxilaryBranchCNN

from pointcloudpyramid import PointCloudPyramid, Pyramid_Layer_1, Pyramid_Layer_2, Pyramid_Layer_3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)


MainBranch = create_RepVGG_A0().to(device)
weightsA0 = torch.load("Pretrained_Networks/RepVGG-A0-train.pth")
MainBranch.load_state_dict(weightsA0)

AuxiliaryBranch = AuxilaryBranchCNN().to(device)
weightsAux = torch.load("Pretrained_Networks/Auxiliary_Network.pth")
AuxiliaryBranch.load_state_dict(weightsAux["model_state_dict"])

PCP = PointCloudPyramid(Pyramid_Layer_1(), Pyramid_Layer_2(), Pyramid_Layer_3()).to(device)

class pointCloudGenerator(nn.Module):
    def __init__(self, mainBranch, auxBranch, pcp):
        super().__init__()
        self.mainBranch = mainBranch
        self.auxBranch = auxBranch
        self.pcp = pcp

    def forward(self, rgbImg, edgeImg):

        x1 = self.mainBranch(rgbImg)
        x2 = self.auxBranch(edgeImg)[0]

        # print(x1.shape)
        # print(x2.shape)

        vec = torch.cat((x1,x2), dim = 1)

        # print(vec.shape)

        pred_PC = self.pcp(vec)

        return pred_PC
    
if __name__ == "__main__":
    # net = pointCloudGenerator(MainBranch, AuxiliaryBranch, PCP)
    print("\n"*2)
    summary(MainBranch, input_size=(32, 3, 128,128))
    print("\n"*2)
    summary(AuxiliaryBranch, input_size=(32, 1, 128,128))
    print("\n"*2)
    summary(PCP, input_size=(32, 1, 2000))

