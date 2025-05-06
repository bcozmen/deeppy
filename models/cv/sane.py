#https://arxiv.org/html/2406.09997v1#bib.bib39
#https://github.com/HSG-AIML/SANE

import torch
import torch.nn as nn

from deeppy.utils import print_args

from deeppy.models.network import Network, OrderedPositionalEmbedding, MaskedTransformerEncoder
from deeppy.models.base_model import BaseModel

class DummyPositionalEmbedding(nn.Module):
	print_args = classmethod(print_args)
	dependencies = []
	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()

	def forward(self,x):
		t = x.shape[:2]  + (2,)
		pos = torch.rand(t) 
		return torch.cat((x,pos), dim=2)

class Sane(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]
	optimize_return_labels = []

	def __init__(self, optimizer_params, vocab_size = 289, embed_dim=1024, latent_dim = 128, num_heads=4, num_layers=4, context_size=50, dropout = 0.1, device = None, criterion = nn.MSELoss()):

		super().__init__(device= device, criterion = criterion)
		#Transformer parameters
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.context_size = context_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.dropout = dropout
		
		#Autoencoder
		self.latent_dim = latent_dim
		self.optimizer_params = optimizer_params

		network_params = self.build_transformer()
		self.net = Network(**network_params).to(self.device)
		
		self.nets = [self.net]
		self.params = [network_params]
		self.objects = [criterion]
		self.train()
	
	
	def init_objects(self):
		pass	

	def __call__(self, X):
		X = self.ensure(X)
		return self.net(X)
	def encode(self,X):
		X = self.ensure(X)
		return self.net.encode(X)
	def decode(self,X):
		X = self.ensure(X)
		return self.net.decode(X)


	def optimize(self, X):
		X,y = self.ensure(X)

		

	def test(self, X):
		X,y = self.ensure(X)

	def build_transformer(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, batch_first= True, norm_first = True, dropout=self.dropout)
		decoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, batch_first= True, norm_first = True, dropout=self.dropout)
		
		encoder_params = {
			"blocks":[nn.Linear,DummyPositionalEmbedding, nn.Dropout, MaskedTransformerEncoder, nn.Linear],
			"block_args":[
				{
					"in_features": self.vocab_size,
					"out_features" : self.embed_dim - 2,
				},
				{
					"num_embeddings" : self.embed_dim ,
					"embedding_dim" : self.vocab_size ,
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
			"blocks":[nn.Linear, DummyPositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear],
			"block_args":[
				{
					"in_features": self.latent_dim,
					"out_features" : self.embed_dim-2,
				},
				{	"num_embeddings" : self.embed_dim ,
					"embedding_dim" : self.vocab_size ,
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
					"out_features":self.vocab_size,
				}
			],
		}

		return {
			"arch_params": encoder_params,
			"decoder_params" : decoder_params,
			"task" : "autoencoder",
			"optimizer_params":self.optimizer_params,
		}	
		
		