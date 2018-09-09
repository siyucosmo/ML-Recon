#!/usr/bin/env python

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from data_utils import SimuData, test_prediction, analysis
import sys
#sys.path.insert(0, '/mnt/home/siyuh/Project/Recon/Unet/')
sys.path.insert(0, './Unet')
from uNet import BasicBlock, Lpt2NbodyNet

def get_parser():
	parser = argparse.ArgumentParser(description="U-Net for ZA -> Nbody net")
	parser.add_argument("-c", "--config_file_path", type=str, default='')
	return parser

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	with open(args.config_file_path) as f:
		configs = json.load(f)

	net = Lpt2NbodyNet(BasicBlock)
	net.cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(),lr=configs["net_params"]['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=configs["net_params"]['reg'])

	base_data_path = configs["base_data_path"]
	output_path = configs["output_path"]

	if configs["is_train"]:
		TrainSet = SimuData(base_data_path,
					configs['train']['data_partition']['lIndex'],
					configs['train']['data_partition']['hIndex'],
					configs['train']['data_partition']['aug'])
		ValSet = SimuData(base_data_path,
					configs['val']['data_partition']['lIndex'],
					configs['val']['data_partition']['hIndex'],
					configs['val']['data_partition']['aug'])
		TrainLoader = DataLoader(TrainSet,
					batch_size=configs['train']['batch_size'],
					shuffle=True,
					num_workers=configs['val']['num_workers'])
		ValLoader   = DataLoader(ValSet,
					batch_size=configs['val']['batch_size'],
					shuffle=False,
					num_workers=configs['val']['num_workers'])
	elif configs["is_test"]:
		TestSet = SimuData(base_data_path,
					configs['test']['data_partition']['lIndex'],
					configs['test']['data_partition']['hIndex'],
					configs['test']['data_partition']['aug'])

		TestLoader  = DataLoader(TestSet,
					batch_size=configs['test']['batch_size'],
					shuffle=False,
					num_workers=configs['test']['num_workers'])

	eval_frequency = configs["train"]["eval_frequency"]
	loss_val = []
	loss_train = []
	iterTime = 0
	best_validation_accuracy = 100

	if(configs["is_train"]):
		for _ in range(configs['train']['num_epoches']):
			for t, data in enumerate(TrainLoader, 0):
				start_time = time.time()
				net.train()
				optimizer.zero_grad()
				NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
				Y_pred = net(NetInput)
				loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
				loss_train.append(loss.item())
				loss.backward()
				optimizer.step()
				print (iterTime,loss.item())
				np.savetxt(output_path+'trainLoss.txt',loss_train)
				if(iterTime!=0 and iterTime%eval_frequency ==0):
					net.eval()
					start_time = time.time()
					_loss=0
					for t_val, data in enumerate(ValLoader,0):
						with torch.no_grad():
							NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
							Y_pred = net(NetInput)
							_loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).item()
					loss_val.append(_loss/t_val)
					np.savetxt(output_path+'valLoss.txt',loss_val)
					print ('valid: ' + str(_loss/t_val))
					if( _loss/t_val < best_validation_accuracy):
						torch.save(net,output_path+'BestModel.pt')
					if(iterTime%500==0 and iterTime!=0):
						torch.save(net,output_path+'BestModel'+str(iterTime)+'.pt')
				iterTime+=1
	if(configs["is_test"]):
		test_prediction(output_path,configs["test"]["model"],TestLoader)

	if(configs["is_analysis"]):
		net = torch.load(output_path+'BestModel.pt')
		net.cuda()
		net.eval()
		#np.save('layer1_weight.npy',net.layer1[0].conv1.weight.data.cpu().numpy())
		#analysis(output_path,"BestModel.pt",configs["analysis"]["size"],configs["analysis"]["A"],configs["analysis"]["phi"],configs["analysis"]["k"])
