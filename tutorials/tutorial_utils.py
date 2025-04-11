import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt 
from IPython.display import HTML, Markdown

import sys
import os

# Add the path to the parent module
sys.path.append(os.path.abspath('../..'))
import deeppy as dp

def get_toy_data_and_plot(f):
	x = np.linspace(-2,2)
	X1,X2 = np.meshgrid(x,x)

	X = np.stack((X1.flatten(), X2.flatten())).T

	X = torch.tensor(X, dtype = torch.float32)
	y = f(X[:,0], X[:,1]).reshape(X1.shape).flatten().unsqueeze(1)

	X = X / 4 + 0.5

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




def get_tutorial_params(clss, env, obs, act):
	if clss.__name__ == "DQN":
		return get_dqn_network_params(env,obs,act)

def get_optimizer_params(lr):
	Scheduler_params = {
	"scheduler" : optim.lr_scheduler.StepLR,
	"gamma":0.01**(1/1000),
	"auto_step":False,
	"step_size" : 1
	}

	Optimizer_params = {
	    "optimizer":optim.AdamW,
	    "optimizer_args":{"lr" : lr, "amsgrad" : True},
	    "clipper":torch.nn.utils.clip_grad_value_,
	    "clipper_params":{"clip_value" : 100},
	    "scheduler_params":Scheduler_params,
	}
	return Optimizer_params
def get_epsilon_params(env):
	return {
	"eps":0.9,
	"eps_end":0.05,
	"eps_decay":1000,
	"random_generator":env.action_space.sample,
	}

def get_sac_params(env,obs,act):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	layers = [obs,256,256,act]
	lr = 1e-4

	arch_params = {
	    "layers":layers,
	    "blocks":[nn.Linear, nn.ReLU],
	    "block_args":[],
	    "out_act":nn.Softmax,
	    "out_params":{},
	    "weight_init":None,
	}

	Network_params = {
	    "arch_params" : arch_params,
	    "decoder_params":None,
	    "task":'reg',
	    "optimizer_params":get_optimizer_params(lr),
	}

	SAC_params = {
	    "ddqn_params" : get_ddqn_params(env,obs,act,lr,layers),
	    "pnet_params":Network_params,
	    "alpha_lr":lr,
	    "gamma":0.99,
	    "target_entropy":-1.0,
	    "device": device,
	    "criterion":nn.SmoothL1Loss(),
	    "mode":'discrete',
	}
	return SAC_params





def get_ddqn_params(env,obs,act, l = None, lay = None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	layers = [obs,128,128,act]
	if lay is not None:
		layers = lay

	lr = 1e-3
	if l is not None:
		lr = l

	arch_params = {
	    "layers":layers,
	    "blocks":[nn.Linear, nn.ReLU],
	    "block_args":[],
	    "out_act":nn.Identity,
	    "out_params":{},
	    "weight_init":None,
	}
	
	Network_params = {
	    "arch_params" : [arch_params],
	    "decoder_params":None,
	    "task":'reg',
	    "optimizer_params":get_optimizer_params(lr),
	}
	if l is None:
		DDQN_params = {
		    "network_params" : Network_params,
		    "gamma":0.99,
		    "tau":0.005,
		    "eps_params":get_epsilon_params(env),
		    "device":device,
		    "criterion":nn.SmoothL1Loss(),
		}
	else:
		DDQN_params = {
		    "network_params" : Network_params,
		    "gamma":0.99,
		    "tau":0.005,
		    "eps_params":get_epsilon_params(env),
		}

	return DDQN_params

def get_dqn_params(env,obs,act):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	layers = [obs,128,128,act]
	lr = 5*1e-4
	arch_params = {
	    "layers":layers,
	    "blocks":[nn.Linear, nn.ReLU],
	    "block_args":[],
	    "out_act":nn.Identity,
	    "out_params":{},
	    "weight_init":None,
	}
	Network_params = {
	    "arch_params" : [arch_params],
	    "decoder_params":None,
	    "task":'reg',
	    "optimizer_params":get_optimizer_params(lr),
	}
	DQN_params = {
    "network_params" : Network_params,
    "gamma":0.99,
    "tau":0.005,
    "eps_params":get_epsilon_params(env),
    "variant":'DQN',
    "device":device,
    "criterion":nn.SmoothL1Loss(),
	}

	return DQN_params


def get_rl_tutorial(tut=None):
	tutorial_params = [get_dqn_params, get_ddqn_params, get_sac_params]
	algos = dp.models.rl.algorithms

	if tut is None:
		display(Markdown("Please select your algorithm as input integers"))
		for ix,i in enumerate((algos)):
		    print(f"{ix} - {i}")
		print()
		algo = int(input("\rSelect algorithm : "))
	else:
		algo = tut

	clss = dp.models.rl.classes[algo]
	env_name = dp.models.rl.envs[algo]
	buf_size = dp.models.rl.buffers[algo]
	print(f"Selected algorithm {clss.__name__}")
	print(f"Checkpoint environment : {env_name}")
	print(f"Checkpoint Buffer size  {buf_size}")
	return clss, env_name, buf_size, tutorial_params[algo]
