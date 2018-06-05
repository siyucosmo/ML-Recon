import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class CGAN:
	def __init__(self,
					generator,
					discriminator,
					configs):
		self._generator = generator
		self._discriminator = discriminator
		self._hyper_parameters = configs
		self._conditioning = configs["conditioning"]
		d_config = configs["d"]
		g_config = configs["g"]
		if d_config["optimizer_name"] == "Adam":	
			self._d_optimizer = torch.optim.Adam(self._discriminator.parameters(),
								lr=d_config['learning_rate'],
								betas=d_config['optim_betas'],
								eps=d_config["eps"],
								weight_decay=d_config["weight_decay"]
								)
		elif d_config["optimizer_name"] == "SGD":
			self._d_optimizer = torch.optim.SGD(self._discriminator.parameters(),
								lr=d_config['learning_rate'],
								momentum=d_config['momentum'],
								weight_decay=d_config['weight_decay'])
		else:
			raise Exception("Invalid discriminator config")

		if g_config["optimizer_name"] == "Adam":
			self._g_optimizer = torch.optim.Adam(self._generator.parameters(),
								lr=g_config['learning_rate'],
								betas=g_config['optim_betas'],
								eps=g_config["eps"],
								weight_decay=g_config["weight_decay"])
		elif d_config["optimizer_name"] == "SGD":
			self._d_optimizer = torch.optim.SGD(self._generator.parameters(),
								lr=g_config['learning_rate'],
								momentum=g_config['momentum'],
								weight_decay=g_config['weight_decay'])
		else:
			raise Exception("Invalid generator config")


	def train_discriminator_step(self,
		g_batch_x,
		d_batch_x,
		d_batch_y):
		self._d_optimizer.zero_grad()
		self._discriminator.train()
		criterion = nn.BCELoss()
		real_data = torch.cat((d_batch_x, d_batch_y), dim=1)\
			if self._conditioning\
			else d_batch_y
		real_labels_out = self._discriminator(real_data).view(-1)
		real_loss = criterion(real_labels_out,
						torch.autograd.Variable(torch.ones(d_batch_x.size()[0]),
												requires_grad=False).cuda())
		real_loss.backward()

		fake_data = self._generator(g_batch_x).detach()
		fake_data = torch.cat((g_batch_x, fake_data), dim=1)\
			if self._conditioning\
			else fake_data
		fake_labels_out = self._discriminator(fake_data).view(-1)
		fake_loss = criterion(fake_labels_out,
						torch.autograd.Variable(torch.zeros(g_batch_x.size()[0]),
												requires_grad=False).cuda())
		fake_loss.backward()
		self._d_optimizer.step()
		print ("D:"+str(fake_loss.item()))


	def train_generator_step(self,
								g_batch_x,
								g_batch_y,
								extra_loss=None,
								extra_loss_fraction=0):
		self._g_optimizer.zero_grad()
		self._generator.train()
		generator_out = self._generator(g_batch_x)
		fake_data = torch.cat((g_batch_x, generator_out), 1)\
			if self._conditioning\
			else generator_out
		criterion = nn.BCELoss()
		g_loss = criterion(self._discriminator(fake_data).view(-1),
						torch.autograd.Variable(torch.ones(g_batch_x.size()[0]),
												requires_grad=False).cuda())
		if extra_loss:
			g_loss = (1 - extra_loss_fraction) * g_loss\
					+ extra_loss_fraction * extra_loss(generator_out, g_batch_y)
		g_loss.backward()
		self._g_optimizer.step()
		print ("G:"+str(g_loss.item()))

	def forward_generator(self, g_batch_x):
		return self._generator(g_batch_x)

	def forward_discriminator(self, g_batch_x):
		return self._discriminator(g_batch_x)

	def train(self):
		self._generator.train()
		self._discriminator.train()

	def eval(self):
		self._generator.eval()
		self._discriminator.eval()
