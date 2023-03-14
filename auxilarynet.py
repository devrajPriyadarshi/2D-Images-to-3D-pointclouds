import sys
sys.path.insert(0, 'RepVGG')

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AuxilaryBranchCNN(nn.Module):
	"""Return 2 values, [x, x_train]: x_train is used in training, while x is used for PCP"""
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.AvgPool2d(2, stride = 2)
		
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.AvgPool2d(2, stride = 2)

		self.flat = nn.Flatten()
		self.lin1 = nn.Linear(in_features=32*32*3,out_features=1000)
		self.lin2 = nn.Linear(in_features=1000, out_features=10)

	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = torch.flatten( x, 1)
		x = self.lin1(x)

		x_train = self.lin2(F.relu(x))

		return x, x_train

if __name__ == "__main__":

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	net = AuxilaryBranchCNN().to(device)

	print("\n"*2, "Auxilary Branch Network:", "\n"*2)

	summary(net, input_size=( 1, 1, 128, 128))

	print("\n"*2)