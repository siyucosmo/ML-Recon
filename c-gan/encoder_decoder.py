import torch
import torch.nn as nn
from periodic_padding import periodic_padding_3d
from data_utils import crop_tensor

def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class BasicBlock(nn.Module):
	def __init__(self,inplane,outplane,stride = 1, downsample = None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(outplane,outplane,padding=0)
		self.bn2 = nn.BatchNorm3d(outplane)
		self.downsample = downsample

	def forward(self,x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(residual)
		x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = periodic_padding_3d(out,pad=(1,1,1,1,1,1))
		out = self.conv2(out)
		out = self.bn2(out)
		out += residual
		out = self.relu(out)

		return out

class Lpt2NbodyNet(nn.Module):
	def __init__(self, block, layers,leaky_relu):
		super(Lpt2NbodyNet,self).__init__()
		self.leaky_param = leaky_relu
		self.conv1 = nn.Conv3d(in_channels =3, out_channels = 64 ,kernel_size= 7, stride=1, padding=0,bias=True)
		self.batchnorm1 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.layer1 = self._make_layer(block, 64, 128, layers[0],stride=2,downsample=True)
		self.layer2 = self._make_layer(block,128,256,layers[1],stride=2,downsample=True)
		
		self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=0)
		self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1)
		self.layer3 = self._make_layer(block, 128,128, layers[0])
		self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=0)
		self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.deconv3 = nn.ConvTranspose3d(64,64,1,stride=1,padding=0) 
		self.deconv_batchnorm3 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.deconv4 = nn.ConvTranspose3d(64,3,1,stride=1,padding=0)

	

	def _make_layer(self,block,inplanes,planes,blocks,stride=1,downsample=None):
		if(downsample is not None):
			downsample = nn.Sequential(
						nn.Conv3d(inplanes,planes,kernel_size=1,stride=2,padding=0,bias=True),
						nn.BatchNorm3d(planes),)
		layers = []
		layers.append(block(inplanes,planes,stride=stride,downsample=downsample))
		inplanes = planes
		for i in range(1,blocks):
			layers.append(block(inplanes,planes))
		return nn.Sequential(*layers)

	def forward(self,x):
		x = periodic_padding_3d(x, pad=(3,3,3,3,3,3))
		x = nn.functional.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=self.leaky_param, inplace=True) 
		#x = nn.functional.max_pool3d(x,kernel_size=2)
		x = self.layer1(x)
		x = self.layer2(x)
		x = nn.functional.leaky_relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))),negative_slope=self.leaky_param, inplace=True)
		x = self.layer3(x)
		x = nn.functional.leaky_relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))),negative_slope=self.leaky_param, inplace=True)
		x = nn.functional.leaky_relu(self.deconv_batchnorm3(self.deconv3(x)),negative_slope=self.leaky_param, inplace=True)
		x = self.deconv4(x)
		return x
