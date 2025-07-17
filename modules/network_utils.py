import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


from deeppy.utils import print_args
#Should be more generalized with arguments
class Scheduler():
	"""
	A scheduler wrapper to configure schedulers for Optimizer objects
	"""
	print_args = classmethod(print_args)
	def __init__(self, optimizer, scheduler, auto_step = True, **kwargs):
		self.auto_step = auto_step
		self.scheduler = scheduler(optimizer, **kwargs)
	
	def step(self):
		self.scheduler.step()
#CHECK OPTIMIZER SAVE LOAD IS CORRECT
class Optimizer():
	"""
	A optimizer wrapper.

	It combines nn.optim object with scheduler, gradient clipping and GradScaler if amp is used.

	If the scheduler's auto_step parameter is True, every time optimizer takes a step, scheduler
	takes a step too. Else, scheduler.step should be explicitely called by the user

	As Model a nn.Module object or a list of parameters can be given
	"""
	print_args = classmethod(print_args)
	dependencies = [Scheduler]

	def __init__(self,model, configure_optimizer = None, gradient_accumulation_steps = 1,
			  optimizer = optim.AdamW,  optimizer_args = {}, 
			  clipper = None, clipper_params = {}, 
			  scheduler_params = None):
		
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
		
		self.optimizer_args = optimizer_args
		self.scheduler_params = scheduler_params
		self.clipper = clipper
		self.clipper_params = clipper_params
		self._step_counter = 0
		self._optimizer_steps_counter = 0
		
		self.writer = SummaryWriter(log_dir="logs")

		#Initialize with neutral values and then later in in base model.set_optimizerrs
		self.optimizer.zero_grad(set_to_none=True)
		self.gradient_accumulation_steps = int(gradient_accumulation_steps)
		self.scaler = GradScaler(enabled=False)

		self.scheduler = None
		if scheduler_params is not None:
			self.scheduler = Scheduler(self.optimizer, **scheduler_params) 
	
	def step(self, loss):
		

		# Calculate loss wrt accumulation
		loss = loss / self.gradient_accumulation_steps
		# Compute gradients
		self.scaler.scale(loss).backward()
		self._step_counter += 1
		
		# If gradient accumulation steps is not reached, return False
		if (self._step_counter) % self.gradient_accumulation_steps != 0:
			return False
			
		#If using AMP unscale
		self.scaler.unscale_(self.optimizer) 
		
		#Log before clipping
		self.log()

		#Clip
		if self.clipper is not None:
			if self.nn_model:
				self.clipper(self.model.parameters(), **self.clipper_params)
			else:
				params = [p for group in self.model for p in group["params"]]
				self.clipper(params, **self.clipper_params)
			
		# Optimizer step
		self.scaler.step(self.optimizer)
		self.scaler.update()
		self.optimizer.zero_grad(set_to_none=True)

		# Scheduler step (if auto-stepping)
		if self.scheduler is not None and self.scheduler.auto_step:
			self.scheduler.step()
		
		self._optimizer_steps_counter += 1
		
		return True


	def save_states(self):
		if self.scheduler is None:
			sch = None
		else:
			sch = self.scheduler.scheduler.state_dict()
		return {
			"optimizer" : self.optimizer.state_dict(),
			"clipper" : self.clipper,
			"clipper_params" : self.clipper_params,
			"scheduler" : sch,
			"scaler" : self.scaler,
			"_optimizer_steps_counter" : self._optimizer_steps_counter,
			"_step_counter" : self._step_counter,}

	def load_states(self, dic):
		self.clipper_params = dic["clipper_params"]
		self.clipper = dic["clipper"]

		self.optimizer.load_state_dict(dic["optimizer"])
		self.optimizer.zero_grad(set_to_none=True)
		self._optimizer_steps_counter = dic["_optimizer_steps_counter"]
		self._step_counter = dic["_step_counter"]
		self.scaler = dic["scaler"]
		if self.scheduler is not None:
			self.scheduler.scheduler.load_state_dict(dic["scheduler"])

	def log(self):
		# Calculate gradient norm
		param_norms, grad_norms, vs, ms = [], [], [], []
		for group in self.optimizer.param_groups:
			for p in group["params"]:
				
				state = self.optimizer.state[p]
				
				if 'exp_avg_sq' in state:
					# Log the average effective lr
					v_t = state['exp_avg_sq'].norm(2).item()
					vs.append(v_t)
				if 'exp_avg' in state:
					m_t = state['exp_avg']
					momentum_norm = m_t.norm(2).item()
					ms.append(momentum_norm)
				if p.grad is not None:
					param_norm, grad_norm = p.data.detach().norm(2).item(), p.grad.detach().norm(2).item()
					
					param_norms.append(param_norm)
					grad_norms.append(grad_norm)
		
		tag_prefix = "Optimizer/"

		if len(param_norms) > 0:
			self.writer.add_histogram(tag_prefix + "Param_Norms", torch.tensor(param_norms), self._optimizer_steps_counter )
		if len(grad_norms) > 0:
			self.writer.add_histogram(tag_prefix + "Gradient_Norms", torch.tensor(grad_norms), self._optimizer_steps_counter )
		if len(ms) > 0:
			self.writer.add_histogram(tag_prefix + "First_Moment_Norms", torch.tensor(ms) , self._optimizer_steps_counter)
		if len(vs) > 0:
			self.writer.add_histogram(tag_prefix + "Second_Moment_Norms", torch.tensor(vs) , self._optimizer_steps_counter)
		self.writer.add_scalar(tag_prefix + "Lr", self.scheduler.scheduler.get_last_lr()[0], self._optimizer_steps_counter)




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

			#Go for later blocks
			for block,bargs in zip(blocks[1:], block_args[1:]):
				#If its an activation function
				if block.__name__ in self.activation_names:
					#If it's the last layer, change activation with out_act
					#Explicit is better than implicit
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
				elif block.__name__ == "LayerNorm":
					bargs["normalized_shape"] = out
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
			act_name = "linear"


		if self.weight_init == "uniform":
			inits = [nn.init.xavier_uniform_, nn.init.kaiming_uniform_]
			mode = "fan_in"
		elif self.weight_init == "normal":
			inits = [nn.init.xavier_normal_, nn.init.kaiming_normal_]
			mode = "fan_out"


		if act_name == "sigmoid" or act_name == "tanh":
			inits[0](layer.weight, gain = nn.init.calculate_gain(act_name))
		else:
			inits[1](layer.weight, mode = mode, nonlinearity = act_name, a = n_slope)


		if layer.bias is not None and self.weight_init is not None:
			nn.init.zeros_(layer.bias)


