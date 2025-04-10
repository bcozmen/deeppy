import torch
import torch.nn as nn

from deeppy.utils import print_args
from deeppy.models.network_utils import LayerGenerator, Optimizer


class Network(nn.Module):
	print_args = classmethod(print_args)
	dependencies = [LayerGenerator, Optimizer]
	def __init__(self, arch_params, decoder_params = None, task = "reg", optimizer_params = {}):
		super(Network, self).__init__()

		self.task = task
		if isinstance(arch_params, dict):
			arch_params = [arch_params]
		self.arch_params = arch_params
		self.decoder_params = decoder_params

		self.encoder_len = None
		self.model = self.generate()

		self.optimizer_params = optimizer_params
		self.optimizer = Optimizer(self.model, **self.optimizer_params)
		
	def save_states(self):
		return {
			"net" : self.model.state_dict(),
			"optimizer" : self.optimizer.save_states(),
		}

	def load_states(self, dic):
		self.model.load_state_dict(dic["net"])
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

		return nn.Sequential(*net)

	def forward(self, X):
		logits = self.model(X)

		if not self.training and self.task == "classify":
			return (logits > 0.5).float()
		return logits

	def encode(self, X):	
		if self.task != "autoencoder":
			return None
		return self.model[:self.encoder_len](X)

	def decode(self, X):
		if self.task != "autoencoder":
			return None
		return self.model[self.encoder_len:](X)


	def back_propagate(self, loss):
		self.optimizer.step(loss)

	def scheduler_step(self):
		self.optimizer.scheduler.step()




	


