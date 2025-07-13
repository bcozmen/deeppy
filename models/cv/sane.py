#https://arxiv.org/html/2406.09997v1#bib.bib39
#https://github.com/HSG-AIML/SANE
import itertools
import torch
import torch.nn as nn


from deeppy.utils import print_args

from deeppy import Network, SqueezeLastDimention,SqueezeLastDimention2Inputs, QuaternionLoss, NT_Xent, Optimizer,concatInputs
from deeppy import SaneLinearTokenizerBeforePosition
from deeppy import SaneXYZPositionalEmbedding
from deeppy.models import BaseModel

class concatInputsWithPosition(nn.Module):
	def __init__(self, dim = 0, sequence_length = 1, embed_dim=1, num_inputs = 2):
		super().__init__()
		self.dim = dim
		self.sequence_length = sequence_length
		self.embed_dim = embed_dim
		self.num_inputs = num_inputs

		self.unique_pos = nn.Embedding(sequence_length, embed_dim)
		self.layer_pos = nn.Embedding(num_inputs, embed_dim)

		unique_pos_indices = torch.arange(sequence_length).repeat(num_inputs)
		layer_pos_indices = torch.tensor([i for i in range(num_inputs) for _ in range(sequence_length) ])

		self.register_buffer('unique_pos_indices', unique_pos_indices)
		self.register_buffer('layer_pos_indices', layer_pos_indices)


	def forward(self, X):
		p1,p2 = self.unique_pos(self.unique_pos_indices), self.layer_pos(self.layer_pos_indices)
		X = torch.cat(X, dim=self.dim)
		return X + p1 + p2

class AttentionPooling(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		self.query = nn.Parameter(torch.randn(1, latent_dim)) * 0.02
	def forward(self, z):
		# z: (batch_size, context, latent_dim)
		batch_size, context, latent_dim = z.shape
		q = self.query.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)  # (batch_size, 1, latent_dim)
		attn_scores = torch.bmm(z, q) / torch.sqrt(torch.tensor(latent_dim))  # (batch_size, context, 1)
		attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over context dim
		pooled = torch.bmm(attn_weights.transpose(1, 2), z).squeeze(1)  # (batch_size, latent_dim)
		# Project and normalize
		return pooled

class SaneRotationalHead(nn.Module):
	def __init(self, latent_dim, proj_dim, out_dim):
		self.proj = nn.Sequential(
			nn.Linear(latent_dim*2, proj_dim),
			nn.ReLU(),
			nn.Linear(proj_dim, out_dim)
		)
		self._init_weights()
	def _init_weights(self):
		for m in self.proj:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	def forward(self, X):
		proj = self.proj(torch.cat(X, dim = -1))
		return torch.nn.functional.normalize(proj, dim=-1)
	
class FullAttentionPooling(nn.Module):
	def __init__(self, latent_dim, proj_dim, out_dim):
		super().__init__()
		self.query = nn.Parameter(torch.randn(1, latent_dim))
		self.proj = nn.Sequential(
			nn.Linear(latent_dim, proj_dim),
			nn.ReLU(),
			nn.Linear(proj_dim, out_dim)
		)
		self._init_weights()
	def _init_weights(self):
		# Initialize query parameter (e.g., normal with small std)
		nn.init.normal_(self.query, mean=0.0, std=0.02)
		# Initialize weights of linear layers (Xavier uniform)
		for m in self.proj:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def forward(self, z):
		# z: (batch_size, context, latent_dim)
		batch_size, context, latent_dim = z.shape
		
		# Expand query for batch: (1, latent_dim) -> (batch_size, latent_dim, 1)
		q = self.query.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)  # (batch_size, 1, latent_dim)
		
		# Compute attention scores: (batch_size, context, latent_dim) @ (batch_size, latent_dim, 1) -> (batch_size, context, 1)
		attn_scores = torch.bmm(z, q) / torch.sqrt(torch.tensor(latent_dim))  # (batch_size, context, 1)
		
		attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over context dim
		
		# Weighted sum: (batch_size, 1, context) @ (batch_size, context, latent_dim) -> (batch_size, 1, latent_dim)
		pooled = torch.bmm(attn_weights.transpose(1, 2), z).squeeze(1)  # (batch_size, latent_dim)
		
		# Project and normalize
		return torch.nn.functional.normalize(self.proj(pooled), dim=-1)  # (batch_size, output_dim)
    

class Sane(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]
	optimize_return_labels = ["Loss", "Recon Loss", "NTX Loss", "Rot Loss",  "Latent Norm Loss", "Z distance"]	

	def __init__(self, optimizer_params, max_positions, 
		input_dim= 201, latent_dim = 128, projection_dim = 30, pos_token_size = 10,
		embed_dim=1024, num_heads=4, num_layers=4,  dropout = 0.1, context_size=50, bias = True, 
		gamma = [0.05,0.05], ntx_temp = 0.1, noise_augment = 0,
		device = None, amp = False,torch_compile = False, gpu_prefetch = 1):

		super().__init__(device= device, amp=amp, torch_compile=torch_compile, gpu_prefetch = gpu_prefetch)

		#Init Loss function
		self.ntx_temp = ntx_temp
		self.rot_crit = QuaternionLoss(loss_type='relative')
		self.recon_crit = nn.MSELoss()
		self.ntx_crit = NT_Xent(temp = ntx_temp)
		self.gamma = torch.tensor(gamma).to(device)
		self.pos_token_size = pos_token_size
		self.noise_augment = noise_augment

		
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

	def init_log_names(self):
		self.losses_names = ["Recon", "NTX", "Rotation", "Z Norm"]
		self.metrics_names = [""]

	def encode(self,X):
		return self.autoencoder.encode(X)

	def decode(self,X):
		return self.autoencoder.decode(X)

	def embed(self,X):
		return torch.mean(self.encode(X), dim=1)
	
	def forward(self, X):
		X,p = X
		z = self.autoencoder.encode((X,p))
		zp = self.project(z[:,:-self.pos_token_size, :])
		y = self.autoencoder.decode((z,p))
		return z, y, zp

	def get_loss(self,X):
		x_1, p_1,m_1,r_1, x_2, p_2,m_2,r_2 = X
		r_1, r_2 = self.rot_crit.euler_to_quaternion(r_1), self.rot_crit.euler_to_quaternion(r_2)

		if self.noise_augment > 0:
			x_1_noisy = x_1 * (1.0 + self.noise_augment * torch.randn_like(x_1))
			x_2_noisy = x_2 * (1.0 + self.noise_augment * torch.randn_like(x_2))
		else:
			x_1_noisy, x_2_noisy = x_1,x_2
		z_1, y_1, zp_1 = self((x_1_noisy, p_1))
		z_2, y_2, zp_2 = self((x_2_noisy, p_2))

		#z_1_synth = torch.cat((z_2[:,:-self.pos_token_size, :] , z_1[:,-self.pos_token_size:,:]), dim = 1)
		#z_2_synth = torch.cat((z_1[:,:-self.pos_token_size, :] , z_2[:,-self.pos_token_size:,:]), dim = 1)

		#y_1_synth = self.autoencoder.decode((z_1_synth, p_1))
		#y_2_synth = self.autoencoder.decode((z_2_synth, p_2))

		

		q_pred = self.classify((z_1[:,-1,:], z_2[:,-1,:]))
		
		
		#Compute reconstruction loss
		x = torch.cat([x_1, x_2], dim=0)
		y = torch.cat([y_1, y_2], dim=0)
		m = torch.cat([m_1, m_2], dim=0)
		#y_synth = torch.cat([y_1_synth, y_2_synth], dim = 0)
		recon_loss = self.recon_crit(y*m,x) 
		#synth_recon_loss = self.recon_crit(y_synth*m,x)
		
		#Compute rotation loss
		rot_loss = self.rot_crit(q_pred, r_1, r_2)

		#Compute NTX loss
		ntx_loss = self.ntx_crit(zp_1, zp_2)
		z_l2_loss = (z_1.pow(2).mean() +  z_2.pow(2).mean()) / 2

		#Compute final loss
		z_distance = (z_1[:,:-self.pos_token_size, :] - z_2[:,:-self.pos_token_size, :]).pow(2).mean()
		#loss = (self.gamma[0] * (recon_loss + synth_recon_loss) / 2)  + (self.gamma[1] * ntx_loss) + (self.gamma[2] * rot_loss) + (self.gamma[3] * z_l2_loss)
		loss = (self.gamma[0] * recon_loss )  + (self.gamma[1] * ntx_loss) + (self.gamma[2] * rot_loss) + (self.gamma[3] * z_l2_loss)

		losses =  (loss.item(), recon_loss.item(), ntx_loss.item(), rot_loss.item(), z_l2_loss.item())
		metrics = (z_distance.item())

		return loss, (loss.item(), recon_loss.item(), ntx_loss.item(), rot_loss.item(), z_l2_loss.item(), z_distance.item())

	def back_propagate(self,loss):
		return self.optimizer.step(loss)
	
	

	# =====================================================================
	
	def build_autoencoder(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		decoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		
		blocks = [SaneLinearTokenizerBeforePosition,SaneXYZPositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear]
		encoder_params = {
			"blocks": blocks,
			"block_args":[
				{
					"in_features": self.input_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim,
					"input_dim" : self.input_dim
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
			"blocks": blocks,
			"block_args":[
				{
					"in_features": self.latent_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim,
					"input_dim" : self.input_dim
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
			"layers":[self.latent_dim , self.projection_dim * 2, self.projection_dim],
			"blocks":[nn.Linear, nn.LayerNorm, nn.ReLU],
			"block_args":[{"bias" : self.bias}],
			"out_act": nn.ReLU,
			"weight_init":"uniform",
		}

		arch_params = {
			"blocks":[FullAttentionPooling],
			"block_args" : [
				   {
					"latent_dim" : self.latent_dim,
					"proj_dim" : self.projection_dim,
					"out_dim" : self.projection_dim//2
					},
				   ]
		}

		network_params = {
			"arch_params": [arch_params],
			"torch_compile" : self.torch_compile,
		}
		return Network(**network_params).to(self.device), network_params

	def build_classifier_old(self):
		arch_params1 = {
			"blocks":[SqueezeLastDimention2Inputs],
		}

		arch_params2 = {
			"layers":[self.latent_dim * 2 * self.pos_token_size, self.projection_dim, 4],
			"blocks":[nn.Linear, nn.LayerNorm, nn.ReLU],
			"out_act": nn.Identity,
			"weight_init":"uniform",
		}
		network_params = {
			"arch_params": [arch_params2,arch_params1],
			"torch_compile" : self.torch_compile,
		}

		return Network(**network_params).to(self.device), network_params
	
	def build_classifier_encoder(self):
		encoder_params = {
			"d_model" : self.latent_dim, 
			"nhead"  :  1, 
			"dim_feedforward" : 2* self.latent_dim, 
			"batch_first" :  True,
			"norm_first" : True, 
			"dropout" : self.dropout, 
			"activation" : nn.GELU(),
			"bias" : self.bias
		}

		arch_params = {
			"blocks":[concatInputsWithPosition, nn.TransformerEncoderLayer, FullAttentionPooling],
			"block_args" : [
				{
					"dim":1,
					"sequence_length" : self.pos_token_size,
					"embed_dim" : self.latent_dim,
				},
				encoder_params,
				{
					"latent_dim" : self.latent_dim,
					"proj_dim" : self.projection_dim,
					"out_dim" : 4
				},
			]
		}
		network_params = {
			"arch_params": [arch_params],
			"torch_compile" : self.torch_compile,
		}

		return Network(**network_params).to(self.device), network_params

	def build_classifier(self):
		arch_params1 = {
			"blocks":[SaneRotationalHead],
			"block_args" : [
				   {
					"latent_dim" : self.latent_dim,
					"proj_dim" : self.projection_dim,
					"output_dim" : 4
					}
				   ]
		}
		network_params = {
			"arch_params": [arch_params1],
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
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(   module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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