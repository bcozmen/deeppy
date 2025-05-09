import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from deeppy.utils import print_args
from deeppy.modules.network import Network
from torch.cuda.amp import autocast, GradScaler

class EnsureMeta(type):
    def __new__(cls, name, bases, dct):
        # Define the Transform function inside the metaclass
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
        
        # Add the Transform method to the class being created
        dct['ensure_tensor_device'] = ensure_tensor_device
        dct['ensure'] = ensure
        
        # Now, handle wrapping specific methods to apply the transform
        transform_methods = dct.get('_transform_methods', [])

        for key, value in dct.items():
            if callable(value) and key in transform_methods:
                # Wrap the method to apply Transform to input
                original_func = value
                def wrapped_func(self, X, *args, **kwargs):
                    X = self.ensure(X)  # Apply Transform
                    return original_func(self, X, *args, **kwargs)
                dct[key] = wrapped_func
        
        return super().__new__(cls, name, bases, dct)

class BaseModel(ABC):
	"""
	A Basis Object For models.

	"""
	print_args = classmethod(print_args)
	dependencies = []
	optimize_return_labels = []

	_transform_methods = ["optimize"]

	def __init__(self, device = None, criterion = nn.MSELoss(), amp = False):
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
		self.amp = amp
		self.scaler = None
		if self.amp:
			self.scaler = GradScaler()





	
	@abstractmethod
	def init_objects(self):
		pass

	@abstractmethod
	def __call__(self,X):
		pass

	@abstractmethod
	def optimize(self, X):
		pass

	def optimize_amp(self,X):
		with autocast(device_type='cuda', dtype=torch.float16, enabled = self.amp):
			loss = self.get_loss(self.ensure(X))
		
		self.optimizer_step(loss, self.scaler)

		return loss
	@self.ensure
	def get_loss(self,X):
		pass
	def optimizer_step(self,loss, scaler):
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