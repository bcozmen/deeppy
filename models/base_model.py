import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from deeppy.utils import print_args
from deeppy.modules.network import Network


class BaseModel(ABC):
	"""
	A Basis Object For models.

	"""
	print_args = classmethod(print_args)
	dependencies = []
	optimize_return_labels = []
	def __init__(self, device = None, criterion = nn.MSELoss(), torch_compile = False):
		"""
		Initializes Base model

		Args:
			device (torch.device): Device to be used
			attr2 (type): Description of the second parameter.
		"""
		self.torch_compile = torch_compile
		self.device = device
		self.criterion = criterion
		self.training = True
		self.nets = []
		self.params = []
		self.objects = []

	
	@abstractmethod
	def init_objects(self):
		pass

	@abstractmethod
	def __call__(self,X):
		pass

	@abstractmethod
	def optimize(self, X):
		pass
	@torch.no_grad()
	def test(self, X):
		pass

	def train(self):
		[net.train() for net in self.nets]
		self.training = True
	def eval(self):
		[net.eval() for net in self.nets]
		self.training = False
	
	def ensure_tensor_device(self, X):
		if X is None:
			return X
		if not torch.is_tensor(X):
			X = torch.tensor(X)  # Convert to tensor
		if X.device != self.device:
			X = X.to(self.device,non_blocking=True)
		return X

	def ensure(self, X):
		if isinstance(X,tuple):
			return tuple(map(self.ensure_tensor_device,X))
		else:
			return self.ensure_tensor_device(X)
	def save(self,file_name = None, return_dict=False):
		save_dict = {
			"params" : self.params,
			"nets" : [net.save_states() for net in self.nets],
			"objs" : self.objects
		}
		if return_dict:
			return save_dict
		torch.save(save_dict, file_name + "/checkpoint.pt")

	def last_lr(self):
		return [net.last_lr() for net in self.nets]
	def scheduler_step(self):
		for net in self.nets:
			net.scheduler_step()

	@classmethod
	def load(clss, file_name):
		if isinstance(file_name, dict):
			checkpoint = file_name
		else:
			checkpoint = torch.load(file_name + "/checkpoint.pt", weights_only = False)
		params = checkpoint["params"]
		dicts = checkpoint["nets"]
		objs = checkpoint["objs"]

		instance = clss(*params)

		for net,net_dicts in zip(instance.nets, dicts):
			net.load_states(net_dicts)
		
		instance.objects = objs
		instance.init_objects()
		return instance

	def save_states(self):
		return self.save(return_dict = True)

	def load_states(self, dic):
		params = dic["params"]
		dicts = dic["nets"]
		objs = dic["objs"]
		for net,net_dicts in zip(self.nets, dicts):
			net.load_states(net_dicts)

	



class Model(BaseModel):
	print_args = classmethod(print_args)
	dependencies = [Network]
	optimize_return_labels = ["Loss"]
	def __init__(self, network_params, device = None, criterion = nn.MSELoss()):
		super().__init__(device = device, criterion=criterion)

		self.net = Network(**network_params).to(self.device)

		self.params = [network_params, device]
		self.nets = [self.net]
		self.objects = [criterion]

		self.train()

	def init_objects(self):
		self.criterion = self.objects[0]

	
	def __call__(self,X):
		X = self.ensure(X)
		outs = self.net(X)
		return outs
	
	def optimize(self, X):
		X,y = self.ensure(X)  
		outs = self(X)

		loss = self.criterion(outs,y)
		self.net.back_propagate(loss)

		return loss.item()

	def test(self, X):
		X,y = self.ensure(X)  
		

		with torch.no_grad():
			outs = self(X)
			loss = self.criterion(outs,y)	

		return loss.item()