import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from utils import Scheduler
class LayerGenerator():
	def __init__(self):
		pass


	def generate(self, type,layers = [], decs = [], args = [], hidden_act = nn.ReLU, out_act = nn.ReLU, weight_init = None):
		net = []
		
		if len(layers) > 1:
			for ix, (i_size, o_size) in enumerate(zip(layers[:-1], layers[1:])):
				if ix != len(layers)-2:
					act = hidden_act()
				else:
					act = out_act()  
				
				if len(args) > 0:
					layer = type(i_size, o_size, **(args[0]))
				else:
					layer = type(i_size, o_size)
				
				self.init_weights(layer,act, weight_init=weight_init)
				net.append(layer)

				for dec,arg in zip(decs,args[1:]):
					net.append(dec(**arg))
				net.append(act)
		else:
			if len(args) > 0:
				layer = type(**(args[0]))
			else:
				layer = type()
			net.append(layer)

		return net

	def init_weights(self, layer, act, weight_init):
		if isinstance(act, nn.Sigmoid):
			if weight_init == "uniform":
				nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
			elif weight_init == "normal":
				nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
		elif isinstance(act, nn.ReLU):
			if weight_init == "uniform":
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
			elif weight_init == "normal":
				nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
		elif isinstance(act, nn.LeakyReLU):
			if weight_init == "uniform":
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
			elif weight_init == "normal":
				nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
		else:
			if weight_init == "uniform":
				nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
			elif weight_init == "normal":
				nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='linear')
		if layer.bias is not None and weight_init is not None:
			nn.init.zeros_(layer.bias)

class Network(nn.Module):
	def __init__(self, arch_params, decoder_params = None, task = "reg", criterion = nn.MSELoss, device = None,
					   optimizer = optim.AdamW, lr = 1e-4, weight_decay = 0 , scheduler_params = None):
		super(Network, self).__init__()

		self.layer_generator = LayerGenerator()

		self.task = task
		self.criterion = criterion()

		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device

		net = []
		for param in arch_params:
			net += self.layer_generator.generate(**param)

		if self.task == "autoencoder" and decoder_params is not None:
			self.encoder_len = len(net)
			for param in decoder_params:
				net += self.layer_generator.generate(**param)

		self.model = nn.Sequential(*net)
		self.optimizer = optimizer(self.model.parameters() , lr= lr, weight_decay = weight_decay, amsgrad = True)
		
		self.scheduler = None
		if scheduler_params is not None:
			self.scheduler = Scheduler(self.optimizer, **scheduler_params) 

		self.to(self.device)

	def predict(self, X):
		return self.forward(X)
		
		

	def optimize(self, *X):
		X,y = X
		X,y = map(self.ensure_tensor_device,list(X))
		outputs = self(X)
		
		loss = self.criterion(outputs, y)
		self.back_propagate(loss)
		return loss

	def back_propagate(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		if self.scheduler is not None:
			self.scheduler.step()

	def forward(self, X):
		X = self.ensure_tensor_device(X)
		logits = self.model(X)

		if not self.training and self.task == "classify":
			return (logits > 0.5).float()
		return logits

		
	def ensure_tensor_device(self,x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor if it's not already one
		if x.device != self.device:
			x = x.to(self.device)
		return x

	def encode(self, X):	
		if self.task != "autoencoder":
			return None
		X = self.ensure_tensor_device(X)
		return self.model[:self.encoder_len](X)

	def decode(self, X):
		if self.task != "autoencoder":
			return None
		X = self.ensure_tensor_device(X)
		return self.model[self.encoder_len:](X)


	def test(self, *X):
		X,y = X
		with torch.no_grad():
			X,y = self.ensure_tensor_device(X), self.ensure_tensor_device(y)
			outs = self(X)
			loss = self.criterion(outs,y)
		return loss
