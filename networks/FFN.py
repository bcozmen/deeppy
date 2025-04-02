import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

"""
T
C = Convolution
L = Linear

Batchnorm - no param
Dropout - 1 param
max-pooling - 1 param


A1
hid_act

A2
out_act

Layers

args





"""

class LayerGenerator():
	def __init__(self):
		pass


	def generate(self, layers, type, decs = [], args = [], hidden_act = nn.ReLU, out_act = nn.ReLU, weight_init = None):
		net = []
		
		
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
	def __init__(self, arch_params, task = "reg", criterion = nn.MSELoss, device = None,
					   optimizer = optim.AdamW, lr = 0.01, weight_decay = 0):
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
		self.model = nn.Sequential(*net)
		self.optimizer = optimizer(self.model.parameters() , lr= lr, weight_decay = weight_decay, amsgrad = True)
		self.to(self.device)

	def predict(self, x):
		pass
		
	def optimize(self, *X):
		X, y = X
		outputs = self(X)
		loss = self.criterion(outputs, y)
		self.back_propagate(loss)
		return loss

	def back_propagate(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def forward(self, X):
		X = self.ensure_tensor_device(X)
		logits = self.model(X)

		if not self.training and self.task == "classify":
			return (logits > 0.5).float()
		return logits

	def ensure_tensor(self,x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor if it's not already one
		return x
	def ensure_device(self,x):
		if x.device != self.device:
			x = x.to(self.device)
		return x
	def ensure_tensor_device(self,x):
		x = self.ensure_device(self.ensure_tensor(x))
		return x

	def encode(self, X):
		with torch.no_grad():
			if self.task != "autoencoder":
				return None
			X = self.ensure_tensor_device(X)
			return self.model[:len(self.model)//2](X)





class FFN(nn.Module):
	def __init__(self, layers, dataset = None, task = "reg" , hidden_act = nn.ReLU, out_act = nn.ReLU, criterion = nn.MSELoss, device = None, \
					   optimizer = optim.AdamW, lr = 0.01,  batch_size = 256, 
					   dropout = None, weight_decay = 0, model_final = False, weight_init = "uniform"):
		super(FFN, self).__init__()
		
		self.layers = layers
		self.hidden_act = hidden_act
		self.out_act = out_act
		self.dropout = dropout

		self.task = task
		self.weight_init = weight_init

		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device

		     

		self.create_model()
		
		
		self.optimizer = optimizer(self.model.parameters() , lr= lr, weight_decay = weight_decay, amsgrad = True)

		if dataset is not None:
			self.criterion = criterion()
			self.model_final = model_final
			self.batch_size = batch_size
			self.dataset = dataset
			self.load_data()

		self.to(self.device)
	def forward(self, X):
		X = self.ensure_tensor_device(X)
		logits = self.model(X)

		if not self.training and self.task == "classify":
			return (logits > 0.5).float()
		return logits

	def create_model(self):
		net = self.create_layers(self.layers)
		if self.task == "autoencoder":
			net += self.create_layers(self.layers[::-1])

		self.model = nn.Sequential(*net)
		if self.weight_init is not None:
			self.init_weights()


	def create_layers(self, layers):
		net = []
		

		for ix, (i_size, o_size) in enumerate(zip(layers[:-1], layers[1:])):
			net.append(nn.Linear(i_size, o_size))
			if ix != len(self.layers)-2:
				net.append(self.hidden_act())
				if not self.dropout is None:
					net.append(nn.Dropout(self.dropout))
			else:
				net.append(self.out_act())  
		return net

	def init_weights(self):
		if self.dropout is None:
			l1,l2 = self.model[::2], self.model[1::2]
		else:
			l1,l2 = self.model[::3], self.model[1::3]
		for layer, act in zip(l1,l2):
			if isinstance(act, nn.Sigmoid):
				if self.weight_init == "uniform":
					nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
				elif self.weight_init == "normal":
					nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)
			elif isinstance(act, nn.ReLU):
				if self.weight_init == "uniform":
					nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
				elif self.weight_init == "normal":
					nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)
			elif isinstance(act, nn.LeakyReLU):
				if self.weight_init == "uniform":
					nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
				elif self.weight_init == "normal":
					nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)
			else:
				if self.weight_init == "uniform":
					nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='linear')
				elif self.weight_init == "normal":
					nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='linear')
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)
			

	

	def load_data(self):
		if self.task == "autoencoder":
			self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
			self.test_loader = DataLoader(self.dataset, batch_size=2048, shuffle=True)
		else:
			train_size = int(0.8 * len(self.dataset))
			test_size = len(self.dataset) - train_size
			self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
			self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
			self.test_loader = DataLoader(self.test_dataset, batch_size=2048, shuffle=True)
		if self.model_final:
			self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
			self.test_loader = DataLoader(self.dataset, batch_size=2048, shuffle=True)

	def get_train_data(self):
		X,y = next(iter(self.train_loader))
		return self.ensure_tensor_device(X), self.ensure_tensor_device(y)
	def get_test_data(self):
		X,y = next(iter(self.test_loader))
		return self.ensure_tensor_device(X), self.ensure_tensor_device(y)

	def ensure_tensor(self,x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor if it's not already one
		return x
	def ensure_device(self,x):
		if x.device != self.device:
			x = x.to(self.device)
		return x
	def ensure_tensor_device(self,x):
		x = self.ensure_device(self.ensure_tensor(x))
		return x

	def encode(self, X):
		with torch.no_grad():
			if self.task != "autoencoder":
				return None
			X = self.ensure_tensor_device(X)
			return self.model[:len(self.model)//2](X)
	def train_epoch(self):
		X,y = self.get_train_data()
		self.train()
		self.optimizer.zero_grad()
		outputs = self(X)
		loss = self.criterion(outputs, y)
		loss.backward()
		self.optimizer.step()
		return loss

	def test(self):
		self.eval()
		with torch.no_grad():
			#X,y = self.train_dataset.dataset.X, self.train_dataset.dataset.y.view(-1,1)
			#X,y = self.ensure_tensor_device(X), self.ensure_tensor_device(y)
			X,y = self.get_test_data()
			outs = self(X)
			loss = self.criterion(outs,y)

			
		return loss
	def back_propagate(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
