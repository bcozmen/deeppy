import torch
import torch.nn as nn

from abc import ABC, abstractmethod,ABCMeta

from deeppy.utils import print_args
from deeppy.modules.network import Network
from torch.amp import GradScaler

class ClassMeta(type):
	def __call__(cls, *args, **kwargs):
		# Create instance without running __init__
		obj = cls.__new__(cls, *args, **kwargs)

		# Now call the actual __init__ from the child
		cls.__init__(obj, *args, **kwargs)

		
		# Call base's after_init (if exists)
		base_after_init = getattr(super(cls, obj), "after_init", None)
		if callable(base_after_init):
			base_after_init()

		transform_methods = getattr(obj, '_to_device_methods', [])
		for name in transform_methods:
			method = getattr(obj, name, None)
			if callable(method):
				def make_wrapper(func):
					def wrapped_func(self, X, *a, **kw):
						X = self.ensure(X)
						return func(X, *a, **kw)
					return wrapped_func
				wrapped = make_wrapper(method).__get__(obj)
				setattr(obj, name, wrapped)
		return obj


class CombinedMeta(ClassMeta, ABCMeta):
	pass

class BaseModel(ABC, metaclass=CombinedMeta):
	"""
	A Basis Object For models.

	"""
	print_args = classmethod(print_args)
	dependencies = []
	optimize_return_labels = []

	_to_device_methods = ["forward", "encode", "decode", "embed", "get_loss", "get_action", "optimize"]
	def after_init(self):
		self.train()
		self.set_optimizers()
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
		self.scaler = GradScaler(enabled=self.amp)
		self.optimizers = None

	def __call__(self, X):
		return self.forward(X)

	def __str__(self):
		return "\n=======================================\n".join([net.__str__() for net in self.nets])

	def init_objects(self):
		self.criterion = self.objects[0]

	def optimize(self,X):
		with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = self.amp):
			loss, return_loss = self.get_loss(X)
		
		self.back_propagate(loss)

		return return_loss
	
	def test(self,X):
		with torch.no_grad():
			with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = self.amp):
				loss, return_loss = self.get_loss(self.ensure(X))
		return return_loss
	
	@abstractmethod
	def get_loss(self,X):
		pass
	@abstractmethod
	def back_propagate(self,loss, scaler):
		pass


	#==========================================================================================
	def ensure_tensor_device(self, X):
		if X is None:
			return X
		if not torch.is_tensor(X):
			X = torch.tensor(X)
		if X.device != self.device:
			X = X.to(self.device, non_blocking=True)
		return X
	def ensure(self, X):
		if isinstance(X, tuple):
			return tuple(map(self.ensure_tensor_device, X))
		else:
			return self.ensure_tensor_device(X)

	def train(self):
		[net.train() for net in self.nets]
		self.training = True
	def eval(self):
		[net.eval() for net in self.nets]
		self.training = False
	def set_optimizers(self):
		if self.optimizers is None:
			self.optimizers = [net.optimizer for net in self.nets]
		for opt in self.optimizers:
			opt.scaler = self.scaler    

	def last_lr(self):
		return [net.last_lr() for net in self.nets]
	def scheduler_step(self):
		for net in self.nets:
			net.scheduler_step()

	def save(self,file_name = None, return_dict=False):
		save_dict = {
			"params" : self.params,
			"nets" : [net.save_states() for net in self.nets],
			"objs" : self.objects
		}
		if return_dict:
			return save_dict
		torch.save(save_dict, file_name + "/checkpoint.pt")
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
	def __init__(self, network_params, device = None, criterion = nn.MSELoss(), amp=True):
		super().__init__(device = device, criterion=criterion, amp=amp)

		self.net = Network(**network_params).to(self.device)
		self.params = [network_params, device]
		self.nets = [self.net]
		self.objects = [criterion]


	def forward(self,X):
		return self.net(X)
	
	def get_loss(self,X):
		X,y= X
		outs = self(X)

		loss = self.criterion(outs,y)
		return loss, loss.item()

	def back_propagate(self,loss):
		self.net.back_propagate(loss)

	