import torch
import torch.nn as nn

from deeppy.utils import print_args
from deeppy.modules.network_utils import LayerGenerator, Optimizer


class Network(nn.Module):
	print_args = classmethod(print_args)
	dependencies = [LayerGenerator, Optimizer]
	def __init__(self, arch_params, decoder_params = None, task = "reg", optimizer_params = None, torch_compile = False):
		super(Network, self).__init__()

		self.torch_compile = torch_compile
		self.task = task
		if isinstance(arch_params, dict):
			arch_params = [arch_params]
		if isinstance(decoder_params, dict):
			decoder_params = [decoder_params]
		self.arch_params = arch_params
		self.decoder_params = decoder_params

		
		self.generate()

		self.optimizer_params = optimizer_params
		if self.optimizer_params is not None:
			self.optimizer = Optimizer(self.model, **self.optimizer_params)
		else:
			self.optimizer = None
		
	def save_states(self):
		try:
			optimizer_dict = self.optimizer.save_states(),
		except:
			optimizer_dict = None
		return {
			"net" : self.model.state_dict(),
			"optimizer" : optimizer_dict
		}

	def load_states(self, dic):
		self.model.load_state_dict(dic["net"])
		if self.optimizer is not None:
			self.optimizer.load_states(dic["optimizer"])

	def generate(self):
		layer_generator = LayerGenerator()
		net = []

		for param in self.arch_params:
			net += layer_generator.generate(**param)

		if self.task == "autoencoder" and self.decoder_params is not None:
			self.encoder_len = len(net)
			for param in self.decoder_params:
				net += layer_generator.generate(**param)
		

		self.model = nn.Sequential(*net)

		if self.task == "autoencoder":
			self.encode = self.model[:self.encoder_len]
			self.decode = self.model[self.encoder_len:]

			if self.torch_compile:
				self.encode, self.decode = torch.compile(self.encode), torch.compile(self.decode)
		elif self.torch_compile:
			self.model = torch.compile(self.model)	


	def forward(self, X):
		if self.task == "autoencoder":
			return self.decode(self.encode(X))

		logits = self.model(X)

		if not self.training and self.task == "classify":
			return (logits > 0.5).float()
		return logits


	def back_propagate(self, loss):
		self.optimizer.step(loss)

	def scheduler_step(self):
		self.optimizer.scheduler.step()
	def last_lr(self):
		return self.optimizer.scheduler.scheduler.get_last_lr()[0]




	


