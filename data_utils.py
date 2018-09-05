import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import sys
sys.path.insert(0,'/mnt/home/siyuh/Project/Recon/analysis')
from genwaves import genwaves

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
			LPT = swap(LPT,0,1)
			Nbody = np.transpose(Nbody, axes = (0,2,3,1))
			Nbody = swap(Nbody,0,2)
			Nbody = swap(Nbody,0,1)
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


#def crop_tensor(x):
#	return x.narrow(2,1,x.shape[2]-1).narrow(3,1,x.shape[3]-1).narrow(4,1,x.shape[4]-1).contiguous()
def crop_tensor(x):
	x = x.narrow(2,1,x.shape[2]-3).narrow(3,1,x.shape[3]-3).narrow(4,1,x.shape[4]-3).contiguous()
	return x

def test_prediction(path,model,TestLoader):
	net = torch.load(model)
	net.cuda()
	net.eval()
	
	for t, data in enumerate(TestLoader, 0):
		print (t)
		NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
		Y_pred = net(NetInput)
		np.save(path+'test_'+str(t)+'.npy',np.concatenate((np.squeeze(Y_pred.data.cpu().numpy()),np.squeeze(data[1].numpy())),axis=0))
	
	#data = np.fromfile('/mnt/home/siyuh/Project/Recon/data/data_version3/32-pancake/00-00-03-phi090/00000000-00001000.32.10.f4',dtype='f4').reshape([-1,32,32,32,10])
	#data = np.fromfile('/mnt/home/siyuh/Project/Recon/data/data_version3/32-pancake/01.80/00000000-00001000.32.10.f4',dtype='f4').reshape([-1,32,32,32,10])
	#data = np.fromfile('/mnt/home/siyuh/Project/Recon/data/data_version3/32-pancake/Om/om-0.32-00000000-00001000.32.10.f4',dtype='f4').reshape([-1,32,32,32,10])
	#data = np.fromfile('/mnt/home/siyuh/Project/Recon/data/data_version3/32-pancake/32-sm/sm-24.00/00000000-00001000.32.10.f4',dtype='f4').reshape([-1,32,32,32,10])
	#data = np.fromfile('/mnt/home/siyuh/Project/Recon/data/data_version3/32-pancake/dual_pancake/00-00-03-phi090-00-00-10-phi090/00000000-00001000.32.10.f4',dtype='f4').reshape([-1,32,32,32,10])
	#for t in range(0,1):
	#	print (t)
	#	data_temp = torch.autograd.Variable(torch.from_numpy(np.expand_dims(np.einsum('ijkl->lijk', data[t][:,:,:,1:4]),axis=0)).float(),requires_grad=False).cuda()
	#	NetInput = torch.autograd.Variable(data_temp,requires_grad=False).cuda()
	#	Y_pred = net(NetInput)
	#	#np.save(path+'pancake_00-00-03-test_'+str(t)+'.npy',np.squeeze(Y_pred.data.cpu().numpy()))
	#	#np.save(path+'01.80_test_'+str(t)+'.npy',np.squeeze(Y_pred.data.cpu().numpy()))
	#	#np.save(path+'sm/sm_24_test_'+str(t)+'.npy',np.squeeze(Y_pred.data.cpu().numpy()))
	#	np.save(path+'dual_pancake/k-00-00-03_k-00-00-10_'+str(t)+'.npy',np.squeeze(Y_pred.data.cpu().numpy()))
	

def analysis(path,model,size, A, phi, k):
        #data = genwaves(size, A, phi, k)
        #data = np.einsum('ijkl->lijk',data)
        data = np.zeros([3,32,32,32])
        data = np.expand_dims(data,axis=0)
        NetInput = torch.autograd.Variable(torch.from_numpy(data).float(),requires_grad=False).cuda()
        net = torch.load(path+model)
        net.cuda()
        net.eval()
        Y_pred = net(NetInput)
        print (size,A,phi,k)
        np.save(path+'A_'+str(A)+'_k_'+str(k).replace(" ","")+'_phi_'+str(phi)+'.npy',np.concatenate((np.squeeze(Y_pred.data.cpu().numpy()),np.squeeze(data)),axis=0))
