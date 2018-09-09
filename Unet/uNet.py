import torch
import torch.nn as nn
from periodic_padding import periodic_padding_3d
from data_utils import crop_tensor
import numpy as np

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class BasicBlock(nn.Module):
	def __init__(self,inplane,outplane,stride = 1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		return out

class Lpt2NbodyNet(nn.Module):
	def __init__(self, block):
		super(Lpt2NbodyNet,self).__init__()
		self.layer1 = self._make_layer(block, 3, 64, blocks=2,stride=1)
		self.layer2 = self._make_layer(block,64,128, blocks=1,stride=2)
		self.layer3 = self._make_layer(block,128,128,blocks=2,stride=1)
		self.layer4 = self._make_layer(block,128,256,blocks=1,stride=2)
		self.layer5 = self._make_layer(block,256,256,blocks=2,stride=1)
		self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=0)
		self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1)
		self.layer6 = self._make_layer(block,256,128,blocks=2,stride=1)
		self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=0)
		self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.layer7 = self._make_layer(block,128,64,blocks=2,stride=1)
		self.deconv4 = nn.ConvTranspose3d(64,3,1,stride=1,padding=0)



	def _make_layer(self,block,inplanes,outplanes,blocks,stride=1):
		layers = []
		for i in range(0,blocks):
			layers.append(block(inplanes,outplanes,stride=stride))
			inplanes = outplanes
		return nn.Sequential(*layers)

	def forward(self,x):
		x1 = self.layer1(x)
		x  = self.layer2(x1)
		x2 = self.layer3(x)
		x  = self.layer4(x2)
		x  = self.layer5(x)
		x  = periodic_padding_3d(x,pad=(0,1,0,1,0,1))
		x  = nn.functional.relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))),inplace=True)
		x  = torch.cat((x,x2),dim=1)
		x  = self.layer6(x)
		x  = periodic_padding_3d(x,pad=(0,1,0,1,0,1))
		x  = nn.functional.relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))),inplace=True)
		x  = torch.cat((x,x1),dim=1)
		x  = self.layer7(x)
		x  = self.deconv4(x)

		return x
