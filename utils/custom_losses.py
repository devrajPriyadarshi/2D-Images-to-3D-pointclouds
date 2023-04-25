import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

from chamferdist import ChamferDistance
from geomloss import SamplesLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TotalLoss(nn.Module):
    def __init__(self, a = 1, b = 0, g = 0):
        super(TotalLoss, self).__init__()

        self.alpha = a
        self.beta = b
        self.gamma = g
        self.CD = ChamferDistance()
        # self.CD = SamplesLoss(loss='hausdorff', p = 2)
        # self.EMD = SamplesLoss(loss='sinkhorn', p = 2, debias=False, scaling=0.999, blur = 1e-3)
        self.EMD = SamplesLoss(loss='sinkhorn', p = 1)
        self.PL = None
    
    def forward(self, output, target):
        num_point = output.shape[1]
        chamferLoss = self.CD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))/num_point
        emdLoss = self.EMD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))*num_point
        # projLoss = self.PL(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))
        # print("ChfLoss: ", chamferLoss)
        # print(chamferLoss.shape)
        # print("EMDLoss: ", emdLoss)
        # print(emdLoss.shape)
        total_loss = self.alpha*chamferLoss + self.beta*2*emdLoss #+ self.gamma*projLoss
        # total_loss = self.beta*emdLoss #+ self.gamma*projLoss

        return total_loss