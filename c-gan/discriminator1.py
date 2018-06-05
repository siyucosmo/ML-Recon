import torch
import torch.nn as nn
from encoder_decoder import BasicBlock
from periodic_padding import periodic_padding_3d

class LPT2NbodyDiscriminator(nn.Module):
	def __init__(self, in_channels):
		super(LPT2NbodyDiscriminator, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels = 64 ,kernel_size= 3, stride=2, padding=0,bias=True)
		self.batchnorm1 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.conv2 = nn.Conv3d(in_channels=64, out_channels = 128 ,kernel_size= 3, stride=2, padding=0,bias=True)
		self.batchnorm2 = nn.BatchNorm3d(num_features = 128,momentum=0.1)
		self.conv3 = nn.Conv3d(in_channels=128, out_channels = 256 ,kernel_size= 3, stride=2, padding=0,bias=True)
		self.batchnorm3 = nn.BatchNorm3d(num_features = 256,momentum=0.1)
		self.fc = nn.Linear(4*4*4*256, 1, bias=True);


	def forward(self, x):
		x = periodic_padding_3d(x, pad=(1,1,1,1,1,1))
		x = nn.functional.relu(self.batchnorm1(self.conv1(x)),inplace=True) 
		#x = nn.functional.max_pool3d(x,kernel_size=2)
		x = periodic_padding_3d(x, pad=(1,1,1,1,1,1))
		x = nn.functional.relu(self.batchnorm2(self.conv2(x)),inplace=True)
		x = periodic_padding_3d(x, pad=(1,1,1,1,1,1))
		x = nn.functional.relu(self.batchnorm3(self.conv3(x)),inplace=True)
		x = x.view(x.size()[0], -1)
		x = self.fc(x)
		x = nn.functional.sigmoid(x)
		return x
