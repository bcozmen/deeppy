#https://arxiv.org/pdf/2003.08934
import types

import torch
import torch.nn as nn

from deeppy.models.base_model import BaseModel
from deeppy.modules.network import Network

class Nerf(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]
	optimize_return_labels = ["Loss coarse", "Loss fine"]

	def new_forward(self,X):
		def encode_high_frequency(self,X):
			return X
		X, D = encode_high_frequency(X[:,:3]), encode_high_frequency(X[:,3:])

		L1 = self.partial_forward(X,0)
		L2 = self.partial_forward(torch.cat((X,L1),dim=1),1)

		return self.partial_forward(torch.cat((L2,D), dim=1),2), L2[:,0]

	def __init__(self, network_params,   device = None, criterion = nn.MSELoss(reduction = "none")):

		super().__init__(device= device, criterion = criterion)
		self.network_params = network_params
		
		self.net_coarse = Network(**network_params).to(self.device)
		self.net_coarse.forward = types.MethodType(new_forward, self.net_coarse)
		self.net_fine = Network(**network_params).to(self.device)
		self.net_fine.forward = types.MethodType(new_forward, self.net_fine)

		self.params = [network_params, device, criterion ]
		
		self.nets = [self.net_coarse, self.net_fine]
		self.train()
	
	
	def predict(self, X):
		X = self.ensure(X)
		return self.net_fine(X)

	def sample_uniform(self,tn,tf, N):
		bin_edges = torch.linspace(tn, tf, N + 1, device= self.device)

		u = torch.rand(N, device=self.device)
		lower,upper = bin_edges[:-1], bin_edges[1:]

		return lower + (upper - lower) * u

	def sample_informed(self,w):
		return sample
	def optimize(self, X):
		X,y = self.ensure(X)

		#seperate position and direction
		x,d = X[:,:3], X[:,3:]

		#Sample uniformly
		t = self.sample_uniform(0,1,10)
		r = x + d*t

		#Calculate volume expectation
		X_sampled = torch.cat(r,d)
		rgb_coarsed, sigma = self.net_coarse(X_sampled)

		#Get informat sample
		X_inform_sampled = self.sample_informed(sigma)
		rgb_fine,sigma = self.net_fine(X_inform_sampled)

		#Calculate loss
		loss = self.criterion(rgb_coarsed, y) + self.criterion(rgb_fine, y)

		for net in self.nets:
			net.optimizer.optimizer.zero_grad()
		loss.backward()
		for net in self.nets:
			net.back_propagate(loss=None)
		return loss.item()

	def test(self, X):
		X,y = self.ensure(X)
		
		with torch.no_grad():
			y_pred, mu, logvar = self.predict(X)
		
			con_loss = self.criterion(y_pred, y)  
			kl_loss = self.beta * self.kl_loss(mu, logvar)
			loss = con_loss + kl_loss

		return loss.item(), con_loss.item(), kl_loss.item()
	


	
