import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt 
from IPython.display import HTML, Markdown
from gymnasium.wrappers import TransformReward
import gymnasium as gym
import sys
import os

# Add the path to the parent module
sys.path.append(os.path.abspath('../../..'))
import deeppy as dp




def get_toy_data_and_plot(is_plot=True, norm =0):
	def f(X,Y):
		return np.sin(X)*np.cos(Y) + np.cos(X)*np.sin(Y)
	
	x = np.linspace(-2,2)
	X1,X2 = np.meshgrid(x,x)

	X = np.stack((X1.flatten(), X2.flatten())).T

	X = torch.tensor(X, dtype = torch.float32)
	y = f(X[:,0], X[:,1]).reshape(X1.shape).flatten().unsqueeze(1)

	if norm == 0:
		X = X / 4 + 0.5
	else:
		X = X/2	

	if is_plot:
		print(f"X shape : {X.shape}")
		print(f"y shape : {y.shape}")

		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(elev=45, azim=135)  # You can tweak these values
		# Plot the surface
		surf = ax.plot_surface(X[:,0].reshape(X1.shape), X[:,1].reshape(X1.shape), y.reshape(X1.shape))

		# Add labels and title
		ax.set_title("3D Plot of f(X, Y) = sin(X)cos(Y) + cos(X)sin(Y)")
		ax.set_xlabel("X1")
		ax.set_ylabel("X2")
		ax.set_zlabel("f(X, Y)")

		# Add color bar

		# Show plot
		plt.show()

	return X,y

def get_toy_dataset():
	X,y = get_toy_data_and_plot(is_plot=False)
	DataGetter_params = {
	    "X" : X,
	    "y": y,
	    "X_valid":None,
	    "test_size":0.,
	    "test_batch_size":None,
	    "valid_batch_size":None,
	    "batch_size":128,
	    "num_workers":0,
	    "pin_memory":True,
	    "shuffle":True,
	}

	return dp.data.DataGetter(**DataGetter_params)


class RLIntroduction():
	def __init__(self, tut=None):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.tutorial_params = [self.get_dqn_params, self.get_ddqn_params, self.get_sac_discrete_params, self.get_sac_cont_params]
		algos = dp.models.rl.algorithms

		if tut is None:
			display(Markdown("Please select your algorithm as input integers"))
			for ix,i in enumerate((algos)):
			    print(f"{ix} - {i}")
			print()
			algo = int(input("\rSelect algorithm : "))
		else:
			algo = tut
		self.algo_ix = algo

		self.clss = dp.models.rl.classes[algo]
		self.env_name = dp.models.rl.envs[algo]
		self.buf_size = dp.models.rl.buffers[algo]
		self.env =  gym.make(self.env_name, render_mode="rgb_array")
		if tut == 3:
			self.env = TransformReward(self.env, lambda r: 5*r)

		try:
			self.obs, self.act = self.env.observation_space.shape[-1], self.env.action_space.shape[-1]
		except:
			self.obs, self.act = self.env.observation_space.shape[-1], self.env.action_space.n
		

		
		checkpoint_name = self.clss.__name__.lower() + "_" + self.env_name.split("-")[0].lower()
		file_name = "checkpoints/" + checkpoint_name
		self.file_name = file_name
		print(f"Selected algorithm {self.clss.__name__}")
		print(f"Checkpoint environment : {self.env_name}")
		print(f"Checkpoint Buffer size  {self.buf_size}")

	def __call__(self):
		return self.clss, self.env_name, self.buf_size, self.tutorial_params[self.algo_ix](), self.file_name, self.env

	def get_optimizer_params(self,lr):
		Scheduler_params = {
		"scheduler" : optim.lr_scheduler.StepLR,
		"gamma":0.01**(1/1000),
		"auto_step":False,
		"step_size" : 1
		}

		Scheduler_params = {
		"scheduler" : optim.lr_scheduler.CyclicLR,
		"base_lr":1e-5,
		"max_lr":5*1e-4,
		"auto_step":True,
		"step_size_up" : 2000
		}

		Optimizer_params = {
		    "optimizer":optim.AdamW,
		    "optimizer_args":{"lr" : lr, "amsgrad" : True},
		    "clipper":torch.nn.utils.clip_grad_value_,
		    "clipper_params":{"clip_value" : 100},
		    "scheduler_params":Scheduler_params,
		}
		return Optimizer_params
	def get_epsilon_params(self):
		return {
		"eps":0.9,
		"eps_end":0.05,
		"eps_decay":1000,
		"random_generator":self.env.action_space.sample,
		}
	def get_sac_cont_params(self):
		qlayers = [self.obs+self.act,256,256,256,self.act]
		players = [self.obs,256,256,256,self.act*2]
		lr = 3*1e-4 



		parch_params = {
		    "layers":players,
		    "blocks":[nn.Linear, nn.ReLU],
		    "out_act":nn.Identity,
		    "out_params":{"dim":1},
		}

		pNetwork_params = {
		    "arch_params" : parch_params,
		    "decoder_params":None,
		    "task":'reg',
		    "optimizer_params":self.get_optimizer_params(lr),
		}
		SAC_params = {
		    "ddqn_params" : self.get_ddqn_params(lr,qlayers),
		    "pnet_params":pNetwork_params,
		    "alpha_lr":lr,
		    "gamma":0.99,
		    "target_entropy":-6.0,
		    "device": self.device,
		    "criterion":nn.MSELoss(),
		    "mode":'continuous',
		    "continuous_action": nn.functional.tanh,
		}
		return SAC_params
	def get_sac_discrete_params(self):
		layers = [self.obs,128,128,128,self.act]
		lr = 1e-5

		arch_params = {
		    "layers":layers,
		    "blocks":[nn.Linear, nn.ReLU],
		    "out_act":nn.Softmax,
		    "out_params":{"dim":1},
		    "weight_init" : None,
		}

		Network_params = {
		    "arch_params" : arch_params,
		    "optimizer_params":self.get_optimizer_params(lr),
		}

		SAC_params = {
		    "ddqn_params" : self.get_ddqn_params(lr,layers),
		    "pnet_params":Network_params,
		    "alpha_lr":lr,
		    "device": self.device,
		    "criterion":nn.MSELoss(),
		    "target_entropy":-1.0,
		}
		return SAC_params





	def get_ddqn_params(self,l = None, lay = None):
		layers = [self.obs,128,128,self.act]
		if lay is not None:
			layers = lay

		lr = 1e-3
		if l is not None:
			lr = l

		arch_params = {
		    "layers":layers,
		    "blocks":[nn.Linear, nn.ReLU],
		    "out_act":nn.Identity,
		    "weight_init" : None,
		}
		
		Network_params = {
		    "arch_params" : [arch_params],
		    "optimizer_params":self.get_optimizer_params(lr),
		}
		if l is None:
			DDQN_params = {
			    "network_params" : Network_params,
			    "eps_params":self.get_epsilon_params(),
			    "device":self.device,
			    "criterion":nn.SmoothL1Loss(),
			}
		else:
			DDQN_params = {
			    "network_params" : Network_params,
			    "eps_params":self.get_epsilon_params(),
			}

		return DDQN_params

	def get_dqn_params(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		layers = [self.obs,128,128,self.act]
		lr = 3*1e-4
		arch_params = {
		    "layers":layers,
		    "blocks":[nn.Linear, nn.ReLU],
		    "out_act":nn.Identity,
		    "weight_init" : None
		}
		Network_params = {
		    "arch_params" : [arch_params],
		    "optimizer_params":self.get_optimizer_params(lr),
		}
		DQN_params = {
	    "network_params" : Network_params,
	    "eps_params":self.get_epsilon_params(),
	    "device":self.device,
	    "criterion":nn.SmoothL1Loss(),
		}

		return DQN_params



