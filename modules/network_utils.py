import torch
import torch.nn as nn
import torch.optim as optim

from deeppy.utils import print_args
#Should be more generalized with arguments
class Scheduler():
	print_args = classmethod(print_args)
	def __init__(self, optimizer, scheduler, auto_step = True, **kwargs):
		self.auto_step = auto_step
		self.scheduler = scheduler(optimizer, **kwargs)
	
	def step(self):
		self.scheduler.step()
#CHECK OPTIMIZER SAVE LOAD IS CORRECT
class Optimizer():
	print_args = classmethod(print_args)
	dependencies = [Scheduler]

	def __init__(self,model, configure_optimizer = None, optimizer = optim.AdamW, optimizer_args = {}, clipper = None, clipper_params = {}, scheduler_params = None):
		
		if configure_optimizer is not None:
			model, optimizer_args = configure_optimizer(model,optimizer_args)
		#Check if model parameters is given as list grouping
		if isinstance(model, list):
			self.nn_model = False
			self.model = model
			self.optimizer = optimizer(self.model, **optimizer_args)
		elif isinstance(model, nn.Module):
			self.nn_model = True
			self.model = model
			self.optimizer = optimizer(self.model.parameters() , **optimizer_args)
		
		self.clipper = clipper
		self.clipper_params = clipper_params

		self.scheduler = None
		if scheduler_params is not None:
			self.scheduler = Scheduler(self.optimizer, **scheduler_params) 
	
	def step(self,loss = None):
		if loss is not None:
			self.optimizer.zero_grad()
			loss.backward()

		if self.clipper is not None:
			if self.nn_model:
				self.clipper(self.model.parameters(), **self.clipper_params)
			else:
				params = [p for group in self.model for p in group["params"]]
				self.clipper(params, **self.clipper_params)

		self.optimizer.step()

		if self.scheduler is not None and self.scheduler.auto_step:
			self.scheduler.step()


	def save_states(self):
		if self.scheduler is None:
			sch = None
		else:
			sch = self.scheduler.scheduler.state_dict()
		return {
			"optimizer" : self.optimizer.state_dict(),
			"clipper" : self.clipper,
			"clipper_params" : self.clipper_params,
			"scheduler" : sch
		}

	def load_states(self, dic):
		self.clipper_params = dic["clipper_params"]
		self.clipper = dic["clipper"]

		self.optimizer.load_state_dict(dic["optimizer"])
		if self.scheduler is not None:
			self.scheduler.scheduler.load_state_dict(dic["scheduler"])



		

class recurrent_layer_helper(nn.Module):
    def forward(self,x):
        tensor, states = x
        self.states = states
        return tensor



#Should be more generalized with arguments
#Better weight init
class LayerGenerator():
	print_args = classmethod(print_args)
	
	def generate(self, layers = [], blocks = [] ,block_args = [], out_act = nn.Identity,  out_params = {}, weight_init = None):
		if out_act is None and len(layers) > 0 :
			raise ValueError("out_act cannot be none. Please use nn.Identity()")
		self.weight_init = weight_init
		self.activation_names = nn.modules.activation.__all__ + ['Identity']

		net = []


		#Make sure that block_args is same size as blocks
		block_args = block_args + [{} for i in range((len(blocks) - len(block_args)))]


		#Go through layers
		for ix,(inp,out) in enumerate(zip(layers[:-1], layers[1:])):
			#Initialize inp->out layer directly
			block, args = blocks[0], block_args[0]

			layer = block(inp,out,**args)
			net.append(layer)
			if block.__name__ in ["RNN","LSTM", "GRU"]:
				net.append(recurrent_layer_helper())


			
			#Go for later blocks
			for block,bargs in zip(blocks[1:], block_args[1:]):
				#If its an activation function
				if block.__name__ in self.activation_names:
					if ix == len(layers)-2:
						block = out_act
						bargs = out_params
					act = block(**bargs)
					net.append(act)
					if self.weight_init is not None:
						self.init_weights(layer, act)
					continue
				#If its a batch norm layer:
				if block.__name__ in nn.modules.batchnorm.__all__:
					bargs["num_features"] = out
				net.append(block(**bargs))



		#If just one layer is given (like nn.Linear), initialize the out activation function too
		if len(net) == 1:
			block = out_act
			bargs = out_params
			act = block(**bargs)
			net.append(act)
			if self.weight_init is not None:
				self.init_weights(layer, act)


		#If no layers were given
		if len(net) == 0:
			for block,args in zip(blocks, block_args):
				net.append(block(**args))
				


		return net



	def init_weights(self, layer, act):
		n_slope = 0
		act_name = act.__class__.__name__.lower()

		if act_name == "identity":
			act_name = "linear"
		elif act_name == "leakyrelu":
			act_name = "leaky_relu"
			n_slope = act.negative_slope
		elif act_name == "softmax":
			act_name = "relu"


		if self.weight_init == "uniform":
			inits = [nn.init.xavier_uniform_, nn.init.kaiming_uniform_]
			mode = "fan_in"
		elif self.weight_init == "normal":
			inits = [nn.init.xavier_normal_, nn.init.kaiming_normal_]
			mode = "fan_out"

		if act_name =="identity" or "softmax":
			pass
		elif act_name == "sigmoid" or act_name == "tanh":
			inits[0](layer.weight, gain = nn.init.calculate_gain(act_name))
		else:
			inits[1](layer.weight, mode = mode, nonlinearity = act_name, a = n_slope)


		if layer.bias is not None and self.weight_init is not None:
			nn.init.zeros_(layer.bias)


