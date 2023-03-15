import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

from chamferdist import ChamferDistance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

        self.alpha = 1
        self.beta = 0
        self.gamma = 0
        self.chamferDistance = ChamferDistance()
    
    def forward(self, output, target):
        chamferLoss = self.chamferDistance(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))

        total_loss = self.alpha*chamferLoss

        return total_loss