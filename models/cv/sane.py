#https://arxiv.org/html/2406.09997v1#bib.bib39
#https://github.com/HSG-AIML/SANE
import itertools
import torch
import torch.nn as nn

from deeppy.utils import print_args

from deeppy import Network, SanePositionalEmbedding, SqueezeLastDimention, NTXentLoss, NT_Xent, Optimizer
from deeppy.models import BaseModel



def parse_input(func):
	def wrapper(self,X):
		if len(X) == 2:
			X = X + (None,)
		return func(self,X)
	return wrapper

class Sane(BaseModel):
	#kwargs = device, criterion
	dependencies = []
	optimize_return_labels = ["Loss"]

	def __init__(self, optimizer_params, max_positions, 
		input_dim= 201, latent_dim = 128, projection_dim = 30,
		embed_dim=1024, num_heads=4, num_layers=4,  dropout = 0.1, context_size=50, bias = True, 
		device = None, gamma = 0.5, ntx_temp = 0.1):

		super().__init__(device= device)

		#Init Loss function
		self.ntx_temp = ntx_temp
		self.recon_crit = nn.MSELoss()
		self.ntx_crit = NT_Xent(temp = ntx_temp)
		self.gamma = gamma

		#Transformer parameters
		self.max_positions = max_positions
		self.input_dim = input_dim
		self.embed_dim = embed_dim
		self.context_size = context_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.dropout = dropout
		self.bias = bias
		self.projection_dim = projection_dim
		
		#Autoencoder
		self.latent_dim = latent_dim
		self.optimizer_params = optimizer_params

		#Create Networks
		self.autoencoder, self.autoencoder_params = self.build_autoencoder()
		self.project , self.project_params = self.build_projection_head()
		self.optimizer = self.configure_optimizer()
		
		
		self.nets = [self.autoencoder, self.project]
		self.params = [self.autoencoder_params, self.project_params]
		self.objects = [self.recon_crit, self.ntx_crit]
		self.train()
	
	
	def init_objects(self):
		self.recon_crit, self.ntx_crit = self.objects	

	@parse_input
	def __call__(self, X):
		X,p,m = self.ensure(X)

		z = self.encode((X,p,m))
		zp = self.project(z)
		y = self.decode((z,p,m))
		return z, y, zp
	@parse_input
	def encode(self,X):
		X,p,m = self.ensure(X)
		
		X = self.autoencoder.model[0](X)
		X = self.autoencoder.model[1](X,p)
		X = self.autoencoder.model[2](X)
		X = self.autoencoder.model[3](X,m)
		X = self.autoencoder.model[4](X)
		return X
	@parse_input
	def decode(self,X):
		X,p,m = self.ensure(X)

		X = self.autoencoder.model[5](X)
		X = self.autoencoder.model[6](X,p)
		X = self.autoencoder.model[7](X)
		X = self.autoencoder.model[8](X,m)
		X = self.autoencoder.model[9](X)

		return X


	def embed(self,X):
		return torch.mean(self.encode(X), dim=1)

	def optimize(self, X):
		x_i, p_i,m_i, x_j, p_j,m_j = self.ensure(X)

		z_i, y_i, zp_i = self((x_i, p_i))
		z_j, y_j, zp_j = self((x_j, p_j))
		# cat y_i, y_j and x_i, x_j, and m_i, m_j
		x = torch.cat([x_i, x_j], dim=0)
		y = torch.cat([y_i, y_j], dim=0)
		m = torch.cat([m_i, m_j], dim=0)
		# compute loss

		recon_loss = self.recon_crit(y*m,x)
		ntx_loss = self.ntx_crit(zp_i, zp_j)
		
		loss = (self.gamma * ntx_loss) + ((1 - self.gamma) * recon_loss)

		self.optimizer.step(loss)

		return loss
		
	@torch.no_grad()
	def test(self, X):
		x_i, p_i,m_i, x_j, p_j,m_j = self.ensure(X)

		z_i, y_i, zp_i = self((x_i, p_i))
		z_j, y_j, zp_j = self((x_j, p_j))
		# cat y_i, y_j and x_i, x_j, and m_i, m_j
		x = torch.cat([x_i, x_j], dim=0)
		y = torch.cat([y_i, y_j], dim=0)
		m = torch.cat([m_i, m_j], dim=0)
		# compute loss

		recon_loss = self.recon_crit(y*m,x)
		ntx_loss = self.ntx_crit(zp_i, zp_j)
		
		loss = (self.gamma * ntx_loss) + ((1 - self.gamma) * recon_loss)
		return loss
	def build_autoencoder(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		decoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		
		encoder_params = {
			"blocks":[nn.Linear,SanePositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear],
			"block_args":[
				{
					"in_features": self.input_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim
				},
				{
					"p" : self.dropout
				},
				{
					"encoder_layer": encoder,
					"num_layers":self.num_layers,
				},
				{
					"in_features" : self.embed_dim,
					"out_features":self.latent_dim,
				}
			],
		}

		decoder_params = {
			"blocks":[nn.Linear, SanePositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear],
			"block_args":[
				{
					"in_features": self.latent_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim
				},
				{
					"p" : self.dropout
				},
				{
					"encoder_layer":decoder,
					"num_layers":self.num_layers,
				},
				{
					"in_features" : self.embed_dim,
					"out_features":self.input_dim,
				}
			],
		}

		network_params = {
			"arch_params": encoder_params,
			"decoder_params" : decoder_params,
			"task" : "autoencoder",
		}	

		return Network(**network_params).to(self.device), network_params
	def build_projection_head(self):
		arch_params1 = {
			"blocks":[SqueezeLastDimention],
		}
		arch_params2 = {
			"layers":[self.latent_dim * self.context_size, self.projection_dim, self.projection_dim],
			"blocks":[nn.Linear, nn.LayerNorm, nn.ReLU],
			"block_args":[{"bias" : self.bias}, {"normalized_shape":self.projection_dim}],
			"out_act": nn.ReLU,
			"out_params":{},
			"weight_init":"uniform",
		}

		network_params = {
			"arch_params": [arch_params1, arch_params2],
		}
		return Network(**network_params).to(self.device), network_params


	def configure_optimizer(self):
		params = itertools.chain(self.autoencoder.named_parameters(), self.project.named_parameters())
		param_dict = {pn: p for pn, p in params}
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

		optim_groups = [
			{"params": decay_params, "weight_decay": self.optimizer_params["optimizer_args"]["weight_decay"]},
			{"params": nodecay_params, "weight_decay": 0.0},
		]
		del self.optimizer_params["optimizer_args"]["weight_decay"]
		return Optimizer(optim_groups, **self.optimizer_params)
