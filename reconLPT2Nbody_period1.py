import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import time
from periodic_padding import periodic_padding_3d

#param
output_path = '/zfsauton/home/siyuh/Project/Recon/result/period/result2/'
iterTime = 0

class SimuData(Dataset):
	def __init__(self,lIndex,hIndex,aug):
		self.datafiles = []
		for x in np.arange(lIndex,hIndex,1000):
			y = ['/zfsauton/home/siyuh/Project/Recon/new_data/'+str(x)+'_'+str(i)+'.npy' for i in range(1000)]
			self.datafiles+=y
		self.aug=aug
	
	def __getitem__(self, index):
		return get_mini_batch(self.datafiles[index],self.aug)

	def __len__(self):
		return len(self.datafiles)

def get_mini_batch(fname,aug):
	x = np.load(fname)
	x = np.einsum('ijkl->lijk', x)
	LPT = x[1:4]
	Nbody = x[7::]
	if(aug==1):
		if np.random.rand() < .5:
			LPT = LPT[:,::-1,...]
			LPT[0] = -LPT[0]
			Nbody = Nbody[:,::-1,...]
			Nbody[0] = -Nbody[0]
		if np.random.rand() < .5:
			LPT = LPT[:,:,::-1,...]
			LPT[1] = -LPT[1]
			Nbody = Nbody[:,:,::-1,...]
			Nbody[1] = -Nbody[1]
		if np.random.rand() < .5:
			LPT = LPT[:,:,:,::-1]
			LPT[2] = -LPT[2]
			Nbody = Nbody[:,:,:,::-1]
			Nbody[2] = -Nbody[2]
		prand = np.random.rand()
		if prand < 1./6:
			LPT = np.transpose(LPT, axes = (0,2,3,1))
			LPT = swap(LPT,0,2)
			LPT = swap(LPT,1,2)
			Nbody = np.transpose(Nbody, axes = (0,2,3,1))
			Nbody = swap(Nbody,0,2)
			Nbody = swap(Nbody,1,2)
		elif prand < 2./6:
			LPT = np.transpose(LPT, axes = (0,2,1,3))
			LPT = swap(LPT,0,1)
			Nbody = np.transpose(Nbody, axes = (0,2,1,3))
			Nbody = swap(Nbody,0,1)
		elif prand < 3./6:
			LPT = np.transpose(LPT, axes = (0,1,3,2))
			LPT = swap(LPT,1,2)
			Nbody = np.transpose(Nbody, axes = (0,1,3,2))
			Nbody = swap(Nbody,1,2)
		elif prand < 4./6:
			LPT = np.transpose(LPT, axes = (0,3,1,2))
			LPT = swap(LPT,1,2)
			LPT = swap(LPT,0,1)
			Nbody = np.transpose(Nbody, axes = (0,3,1,2))
			Nbody = swap(Nbody,1,2)
			Nbody = swap(Nbody,0,1)
		elif prand < 5./6:
			LPT = np.transpose(LPT, axes = (0,3,2,1))
			LPT = swap(LPT,0,2)
			Nbody = np.transpose(Nbody, axes = (0,3,2,1))
			Nbody = swap(Nbody,0,2)
	return torch.from_numpy(LPT.copy()).float(),torch.from_numpy(Nbody.copy()).float()

def swap(x,index1,index2):
	temp = x[index1].copy()
	x[index1] = x[index2]
	x[index2] = temp
	return x

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
	def __init__(self, block, layers):
		super(Lpt2NbodyNet,self).__init__()
		self.conv1 = nn.Conv3d(in_channels =3, out_channels = 64 ,kernel_size= 7, stride=1, padding=0,bias=True)
		self.batchnorm1 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.layer1 = self._make_layer(block, 64, 128, layers[0],stride=2,downsample=True)
		self.layer2 = self._make_layer(block,128,256,layers[1],stride=2,downsample=True)
		
		self.deconv1 = nn.ConvTranspose3d(256,128,2,stride=2,padding=0)
		self.layer3 = self._make_layer(block, 128,128, layers[0])
		self.deconv2 = nn.ConvTranspose3d(128,64,2,stride=2,padding=0)
		self.deconv3 = nn.ConvTranspose3d(64,64,1,stride=1,padding=0) 
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
		x = nn.functional.leaky_relu(self.batchnorm1(self.conv1(x)), negative_slope=0.01, inplace=True) 
		#x = nn.functional.max_pool3d(x,kernel_size=2)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.deconv1(x)
		x = self.layer3(x)
		x = self.deconv2(x)
		x = self.deconv3(x)
		x = self.deconv4(x)
		return x

net = Lpt2NbodyNet(BasicBlock,[3,3])
net.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
#optimizer = torch.optim.SGD(net.parameters(),lr=1e-3,momentum=0.9,weight_decay=0)

#Data input
TrainSet = SimuData(0,8000,1)
ValSet = SimuData(8000,9000,1)
TestSet = SimuData(9000,10000,0)
TrainLoader = DataLoader(TrainSet, batch_size=50,  shuffle=True, num_workers=4)
ValLoader   = DataLoader(ValSet, batch_size=50,  shuffle=False, num_workers=4)
TestLoader  = DataLoader(TestSet, batch_size=1,  shuffle=False, num_workers=4)	

#check param
val = 1
best_validation_accuracy = 100
loss_train = []
loss_val = []

for epoch in range(200):
	for t, data in enumerate(TrainLoader, 0):
		start_time = time.time()
		net.train()
		optimizer.zero_grad()
		NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
		Y_pred = net(NetInput)
		loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
		loss_train.append(loss.data[0])
		loss.backward()
		optimizer.step()
		print (iterTime,loss.data[0])
		np.savetxt(output_path+'trainLoss.txt',loss_train)
		if(t!=0 and (t+1)%20 ==0 and val == 1):
			net.eval()
			start_time = time.time()
			_loss=0
			for t_val, data in enumerate(ValLoader,0):
				NetInput = torch.autograd.Variable(data[0],requires_grad=False,volatile=True).cuda()
				Y_pred = net(NetInput)
				_loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).data[0]
			loss_val.append(_loss/t_val)		
			np.savetxt(output_path+'valLoss.txt',loss_val)
			print ('valid: ' + str(_loss/t_val))
			if( _loss/t_val < best_validation_accuracy):
				torch.save(net,output_path+'BestModel.pt')
			if(iterTime%500==0 and iterTime!=0):
				torch.save(net,output_path+'BestModel'+str(iterTime)+'.pt')
		iterTime+=1
			


