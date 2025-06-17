import torch
import torch.nn as nn

from abc import ABC, abstractmethod,ABCMeta

from deeppy.utils import print_args
from deeppy.modules.network import Network
from torch.cuda.amp import GradScaler

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
						r = func(X, *a, **kw)
						return r
					return wrapped_func
				wrapped = make_wrapper(method).__get__(obj)
				setattr(obj, name, wrapped)
		return obj


class CombinedMeta(ClassMeta, ABCMeta):
	pass

class BaseModel(ABC, metaclass=CombinedMeta):
	"""
	A Basis Object For models. Implements basic functionalities, optimize-test functions etc

	When a new model is initialized, ensure_tensor_device function is appended before each method 
	listed in _to_device_methods (See ClassMeta). Thus, once the torch.device is set, there is no need
	to explicitely push the data into GPU or tensor form. 

	When a new model is initialized, the function after_init is also called after the __init__ function,
	to ensure that model is in training mode and the optimizers are set correctly
	
	The base model implements optimize and test methods. These methods uses get_loss for forward 
	propagation + calculating the loss and back_propagate to backpropagate. Thus these 2 subfunctions must
	be implemented by the developper of the model. 

	To create a new model, the developper basically should initialize the model correctly and then implement 
	forward, get_loss, and back_propaget functions

	Network objects already by default come with optimizers (should be given as arguments) integrated. 
	However for some implementations (when there are more than 1 Network object), it can be more efficient
	to implement the optimizer in the Model class, as well as configuring them. (See cv/Sane model)

	See class Model for example implementation
	...

	Attributes
	----------
		dependencies : List of classes used
			The classes in dependencies are also printen when print_args is called
		optimize_return_labels : List of strings:
			The names of the losses returned by optimization
		_to_device_methods : List of strings
			Methods that should use ensure tensor device by default
		device : torch device object
			Torch device
       	criterion : nn.Module object
       		A criterion to calculate the loss function (For models with multiple criterions, this can be none)
		training: bool
			Flag to control if the pytorch in eval or training mode
		nets : list
			List to keep pytorch networks
		params : list
			List to keep network parameters
		objects : list
			List to keep objects implementd (like criterion)
		amp : boolean
			If use cuda amp
		scaler : GradScaler
			Necessery implementation for amp functionality
		optimizers : list of nn.optim object
			List of optimizers
	"""
	print_args = classmethod(print_args)
	dependencies = []
	optimize_return_labels = []

	_to_device_methods = ["forward", "encode", "decode", "embed", "get_loss", "get_action", "optimize"]
	def after_init(self):
		self.train()
		self.set_optimizers()
	def __init__(self, device = None, criterion = nn.MSELoss(), amp = False, gradient_accumulation = 1):
		"""
		Initializes Base model
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

		self.gradient_accumulation = gradient_accumulation
		self.epoch = 0
	def __call__(self, X):
		return self.forward(X)

	def __str__(self):
		return "\n=======================================\n".join([net.__str__() for net in self.nets])

	def init_objects(self):
		self.criterion = self.objects[0]

	def optimize(self,X):
		self.train()
		with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = self.amp):
			loss, return_loss = self.get_loss(X)
		
		self.back_propagate(loss)


		return return_loss
	@torch.no_grad()
	def test(self,X):
		self.eval()
		with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = self.amp):
			loss, return_loss = self.get_loss(self.ensure(X))
		return return_loss
	
	@abstractmethod
	def get_loss(self,X):
		pass
	@abstractmethod
	def back_propagate(self,loss, scaler):
		pass
	@abstractmethod
	def forward(self,X):
		pass
	#==========================================================================================
	#BASIC FUNCTIONALITY
	def ensure_tensor_device(self, X):
		#Make sure that data is torch tensors and on the correct device
		if X is None:
			return X
		if not torch.is_tensor(X):
			X = torch.tensor(X)
		if X.device != self.device:
			X = X.to(self.device, non_blocking=True)
		return X
	def ensure(self, X):
		#Helper function for ensure tensor device
		if isinstance(X, tuple):
			return tuple(map(self.ensure_tensor_device, X))
		else:
			return self.ensure_tensor_device(X)

	def train(self):
		#Put the pytorch network in training mode
		[net.train() for net in self.nets]
		self.training = True
	def eval(self):
		#Put the pytorch network in eval mode
		[net.eval() for net in self.nets]
		self.training = False
	def set_optimizers(self):
		#Set the self.optimizer list
		if self.optimizers is None:
			self.optimizers = [net.optimizer for net in self.nets]
		for opt in self.optimizers:
			opt.scaler = self.scaler    

	def last_lr(self):
		#Net the last_lr if a scheduler is used
		return [optimizer.scheduler.scheduler.get_last_lr()[0] for optimizer in self.optimizers]
	def scheduler_step(self):
		#Take a scheduler step
		for net in self.nets:
			net.scheduler_step()

	def save(self,file_name = None, return_dict=False):
		#Save the model given a file name
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
		#Load the model from the class.
		#First initialize a new object, and then load the checkpoint
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
		#Helper function to save
		return self.save(return_dict = True)

	def load_states(self, dic):
		#Helper function to load
		params = dic["params"]
		dicts = dic["nets"]
		objs = dic["objs"]
		for net,net_dicts in zip(self.nets, dicts):
			net.load_states(net_dicts)

	def print_param_count(self):
		[net.print_param_count() for net in self.nets]

	



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

	