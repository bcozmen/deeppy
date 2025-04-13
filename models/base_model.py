import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from deeppy.utils import print_args
from deeppy.models.network import Network


class BaseModel(ABC):
	"""
    A Basis Object For models.

    """
	print_args = classmethod(print_args)
	dependencies = []

	def __init__(self, device = None, criterion = nn.MSELoss()):
		"""
        Initializes Base model

        Args:
            device (torch.device): Device to be used
            attr2 (type): Description of the second parameter.
        """
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
	def predict(self,X):
		pass

	@abstractmethod
	def optimize(self, *X):
		pass

	def test(self, *X):
		pass

	def train(self):
		[net.train() for net in self.nets]
		self.training = True
	def eval(self):
		[net.eval() for net in self.nets]
		self.training = False
	
	def ensure_tensor_device(self, X):
		if not torch.is_tensor(X):
			X = torch.tensor(X)  # Convert to tensor
		if X.device != self.device:
			X = X.to(self.device,non_blocking=True)
		return X

	def ensure(self, *X):
		return map(self.ensure_tensor_device,list(X))

	def save(self,file_name = None, return_dict=False):
		save_dict = {
			"params" : self.params,
			"nets" : [net.save_states() for net in self.nets],
			"objs" : self.objects
		}
		if return_dict:
			return save_dict
		torch.save(save_dict, file_name + "/checkpoint.pt")

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

	def reparametrize(self,latent, get_max = False):
		latent_size = latent.shape[1]//2
		mu, std = latent[:, :latent_size], torch.abs(latent[:, latent_size:])
		std = torch.clamp(std, min = 1e-6, max = 4)
		eps = torch.randn_like(std, device = self.device, dtype = torch.float32)

		z = mu + eps * std

		return z, mu, std



class Model(BaseModel):
	print_args = classmethod(print_args)
	dependencies = [Network]
	def __init__(self, network_params, device = None, criterion = nn.MSELoss()):
		super().__init__(device = device, criterion=criterion)

		self.net = Network(**network_params).to(self.device)
		self.task = self.net.task

		self.params = [network_params, device]
		self.nets = [self.net]
		self.objects = [criterion]

		self.train()

	def init_objects(self):
		self.criterion = self.objects[0]

	
	def predict(self,X):
		data = self.ensure(*[X])
		outs = self.net.forward(X)
		return outs

	
	def optimize(self, *X):
		if self.task == "autoencoder":
			X, = self.ensure(*X)  
			y = X
		else:
			X,y = self.ensure(*X)  
		outs = self.predict(X)

		loss = self.criterion(outs,y)
		self.net.back_propagate(loss)

		return loss.item()

	def test(self, *X):
		if self.task == "autoencoder":
			X, = self.ensure(*X)  
			y = X
		else:
			X,y = self.ensure(*X)  
		

		with torch.no_grad():
			outs = self.predict(X)
			loss = self.criterion(outs,y)	

		return loss.item()