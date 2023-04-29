import numpy as np

import torch
import torch.nn as nn

from chamferdist import ChamferDistance
from geomloss import SamplesLoss

from ProjectionLoss import projectionLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

        self.CD = ChamferDistance()
        # self.CD = SamplesLoss(loss='hausdorff', p = 2)
        # self.EMD = SamplesLoss(loss='sinkhorn', p = 2, debias=False, scaling=0.999, blur = 1e-3)
        self.EMD = SamplesLoss(loss='sinkhorn', p = 1)
        self.PL = projectionLoss()

    
    def forward(self, output, target, a = 1, b = 0, c = 0):
        num_point = output.shape[1]

        total_loss = 0

        if a!=0:
            chamferLoss = self.CD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float), bidirectional=True, reduction="sum")
            print("ch_loss:", chamferLoss)
            total_loss += a*chamferLoss
        
        if b!=0:
            emdLoss = self.EMD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))*num_point
            print("emd_loss:", emdLoss.sum())
            total_loss += b*emdLoss.sum()

        if c!=0:
            projLoss = self.PL(output.to(device="cpu", dtype=torch.float), target.to(device="cpu", dtype=torch.float))
            print("proj_loss:", projLoss)
            total_loss += c*projLoss

        return total_loss
        
        # if a!=0 and b!=0 and c == 0:
        #     chamferLoss = self.CD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))
        #     emdLoss = self.EMD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))*num_point

        #     print("ch_loss:", chamferLoss)
        #     print("emd_loss:", emdLoss)

        #     total_loss = a*chamferLoss.sum() + b*emdLoss.sum() 
        #     return total_loss
        
        # if a!=0 and b == 0 and c == 0:

        #     chamferLoss = self.CD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))
        #     print("ch_loss:", chamferLoss)
        #     total_loss = a*chamferLoss.sum() 
        #     return total_loss
        
        # if a==0 and b!=0 and c == 0:
        #     chamferLoss = self.CD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))
        #     emdLoss = self.EMD(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))*num_point
        #     projLoss = self.PL(output.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float))

        #     print("ch_loss:", chamferLoss)
        #     print("emd_loss:", emdLoss)
        #     print("proj_loss:", projLoss)

        #     total_loss = a*chamferLoss.sum() + b*emdLoss.sum() + c*projLoss
    
        #     return total_loss