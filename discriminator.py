import torch
import torch.nn as nn
from encoder_decoder import BasicBlock
from periodic_padding import periodic_padding_3d

class LPT2NbodyDiscriminator(nn.Module):
	def __init__(self, block, layers, in_channels):
		super(LPT2NbodyDiscriminator, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels = 64 ,kernel_size= 7, stride=1, padding=0,bias=True)
		self.batchnorm1 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.layer1 = self._make_layer(block, 64, 128, layers[0],stride=2, downsample=True)
		self.layer2 = self._make_layer(block, 128, 256, layers[1], stride=2, downsample=True)
		self.layer3 = self._make_layer(block, 256, 512, layers[2], stride=2, downsample=True)
		self.fc = nn.Linear(4*4*4*512, 1, bias=True);

	def _make_layer(self,
					block,
					inplanes,
					planes,
					blocks,
					stride=1,
					downsample=None):
		if(downsample is not None):
			downsample = nn.Sequential(
						nn.Conv3d(inplanes,
									planes,
									kernel_size=1,
									stride=2,
									padding=0,
									bias=True),
						nn.BatchNorm3d(planes),)
		layers = []
		layers.append(block(inplanes,
							planes,
							stride=stride,
							downsample=downsample))
		inplanes = planes
		for i in range(1,blocks):
			layers.append(block(inplanes,planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = periodic_padding_3d(x, pad=(3,3,3,3,3,3))
		x = nn.functional.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=0.01, inplace=True) 
		#x = nn.functional.max_pool3d(x,kernel_size=2)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size()[0], -1)
		x = self.fc(x)
		x = nn.functional.sigmoid(x)
		return x