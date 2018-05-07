import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class SimuData(Dataset):
	def __init__(self,base_path,lIndex,hIndex,aug):
		self.datafiles = []
		for x in np.arange(lIndex,hIndex,1000):
			y = [base_path+str(x)+'_'+str(i)+'.npy' for i in range(1000)]
			self.datafiles+=y
		self.aug=aug
	
	def __getitem__(self, index):
		return get_mini_batch(self.datafiles[index],self.aug)

	def __len__(self):
		return len(self.datafiles)

def swap(x,index1,index2):
	temp = x[index1].copy()
	x[index1] = x[index2]
	x[index2] = temp
	return x

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