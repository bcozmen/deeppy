#https://arxiv.org/html/2406.09997v1#bib.bib39
#https://github.com/HSG-AIML/SANE
import itertools
import torch
import torch.nn as nn


from deeppy.utils import print_args

from deeppy import Network, SqueezeLastDimention, QuaternionLoss, NT_Xent, Optimizer
from deeppy import LinearTokenizerBeforePosition
from deeppy import SaneXYZPositionalEmbedding, SanePositionalEmbedding
from deeppy.models import BaseModel

class Sane(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]
	optimize_return_labels = ["Loss", "Recon Loss", "NTX Loss", "Rot Loss"]

	def __init__(self, optimizer_params, max_positions, 
		input_dim= 201, latent_dim = 128, projection_dim = 30,
		embed_dim=1024, num_heads=4, num_layers=4,  dropout = 0.1, context_size=50, bias = True, 
		gamma = [0.05,0.05], ntx_temp = 0.1,
		device = None, amp = False,torch_compile = False):

		super().__init__(device= device, amp=amp)

		self.torch_compile = torch_compile

		#Init Loss function
		self.ntx_temp = ntx_temp
		self.rot_crit = QuaternionLoss(loss_type='relative')
		self.recon_crit = nn.MSELoss()
		self.ntx_crit = NT_Xent(temp = ntx_temp)
		self.gamma = torch.tensor(gamma).to(device)

		
		#Encoder
		self.input_dim = input_dim
		self.max_positions = max_positions
		self.embed_dim = embed_dim
		#Transformerd
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
		self.classify, self.classify_params = self.build_classifier()
		self.nets = [self.autoencoder, self.project, self.classify]
		
		self.optimizer = self.configure_optimizer()
		self.params = [self.autoencoder_params, self.project_params, self.classify_params]
		self.objects = [self.recon_crit, self.ntx_crit, self.rot_crit]
		self.optimizers = [self.optimizer]

	
	def init_objects(self):
		self.recon_crit, self.ntx_crit, self.rot_crit = self.objects

	def forward(self, X):
		X,p = X
		z = self.autoencoder.encode((X,p))

		#z[:,0,4:] = 0
		zp = self.project(z[:,1:])
		y = self.autoencoder.decode((z,p))
		return z, y, zp

	def encode(self,X):
		return self.autoencoder.encode(X)

	def decode(self,X):
		return self.autoencoder.decode(X)

	def embed(self,X):
		return torch.mean(self.encode(X), dim=1)

	def get_loss(self,X):
		x_1, p_1,m_1,r_1, x_2, p_2,m_2,r_2 = X
		r_1, r_2 = self.rot_crit.euler_to_quaternion(r_1), self.rot_crit.euler_to_quaternion(r_2)

		z_1, y_1, zp_1 = self((x_1, p_1))
		z_2, y_2, zp_2 = self((x_2, p_2))
		
		#Compute reconstruction loss
		x = torch.cat([x_1, x_2], dim=0)
		y = torch.cat([y_1, y_2], dim=0)
		m = torch.cat([m_1, m_2], dim=0)
		recon_loss = self.recon_crit(y*m,x)
		
		#Compute rotation loss
		z_rot1 = self.classify(z_1[:,0,:]) #[B_size x 4]
		z_rot2 = self.classify(z_2[:,0,:]) #[B_size x 4]
		rot_loss = self.rot_crit(z_rot1, z_rot2, r_1, r_2)

		#Compute NTX loss
		ntx_loss = self.ntx_crit(zp_1, zp_2)

		#Compute final loss
		loss = (self.gamma[0] * ntx_loss) + ((torch.tensor(1).to(self.device) - torch.sum(self.gamma)) * recon_loss) + (self.gamma[1] * rot_loss)

		return loss, (loss.item(), recon_loss.item(), ntx_loss.item(), rot_loss.item())

	def back_propagate(self,loss):

		self.optimizer.step(loss)

	# =====================================================================
	
	def build_autoencoder(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		decoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		
		encoder_params = {
			"blocks":[LinearTokenizerBeforePosition,SaneXYZPositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear],
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
			"blocks":[LinearTokenizerBeforePosition, SaneXYZPositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear],
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
			"torch_compile" : self.torch_compile,
		}	

		return Network(**network_params).to(self.device), network_params
	def build_projection_head(self):
		arch_params1 = {
			"blocks":[SqueezeLastDimention],
		}
		arch_params2 = {
			"layers":[self.latent_dim * (self.context_size - 1), self.projection_dim, self.projection_dim],
			"blocks":[nn.Linear, nn.LayerNorm, nn.ReLU],
			"block_args":[{"bias" : self.bias}],
			"out_act": nn.ReLU,
			"weight_init":"uniform",
		}

		network_params = {
			"arch_params": [arch_params1, arch_params2],
			"torch_compile" : self.torch_compile,
		}
		return Network(**network_params).to(self.device), network_params

	def build_classifier(self):
		arch_params = {
			"blocks":[nn.Linear],
			"block_args":[
				{
					"in_features": self.latent_dim,
					"out_features" : 4,
				},
			],
			"out_act": nn.Identity,
			"weight_init":"uniform",
		}
		arch_params = {
			"layers":[self.latent_dim, self.projection_dim, 4],
			"blocks":[nn.Linear, nn.LayerNorm, nn.ReLU],
			"out_act": nn.Identity,
			"weight_init":"uniform",
		}
		network_params = {
			"arch_params": [arch_params],
			"torch_compile" : self.torch_compile,
		}
		return Network(**network_params).to(self.device), network_params
	def configure_optimizer(self):
		params = itertools.chain(*[k.named_parameters() for k in self.nets])
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

	# =====================================================================
	#HELPER FUNCTIONS


	def load(self, file_name):
		#Load the model from the class.
		#First initialize a new object, and then load the checkpoint
		if isinstance(file_name, dict):
			checkpoint = file_name
		else:
			checkpoint = torch.load(file_name + "/checkpoint.pt", weights_only = False)
		
		dicts = checkpoint["nets"]
		objs = checkpoint["objs"]
		optimizer_dicts = checkpoint["optimizer"]
		

		for net,net_dicts in zip(self.nets, dicts):
			net.load_states(net_dicts)
		if optimizer_dicts is not None:
			[optimizer.load_states(dic) for optimizer, dic in zip(self.optimizers, optimizer_dicts)]
		
		self.objects = objs
		self.init_objects()
		return self