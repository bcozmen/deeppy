import torch
import torch.nn as nn

from deeppy import Network, OrderedPositionalEmbedding, MaskedTransformerEncoder
from deeppy.models import BaseModel



class GPT(BaseModel):
	dependencies = [Network]
	optimize_return_labels = ["Loss"]
	#test_return_labels = ["Accuracy"]
	def __init__(self, optimizer_params, vocab_size = 3, embed_dim=48, num_heads=3, num_layers=3, context_size=11, dropout = 0.1, 
		device = None, criterion = nn.CrossEntropyLoss(), torch_compile = False):
		super().__init__(device = device, criterion=criterion)
		self.torch_compile = torch_compile

		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.context_size = context_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.optimizer_params = optimizer_params
		self.dropout = dropout

		network_params = self.build_transformer()
		self.net = Network(**network_params).to(self.device)
		
		self.nets = [self.net]
		self.params = [network_params]
		self.objects = [criterion]
		self.train()
	
	def init_objects():
		pass
	def __call__(self,X):
		X = self.ensure(X)
		outs = self.net(X)
		return outs
	def optimize(self, X):
		X,y = self.ensure(X)  
		outs = self(X)

		loss = self.criterion(outs.view(-1, outs.size(-1)),y.view(-1))
		self.net.back_propagate(loss)

		return loss.item()
	@torch.no_grad()
	def test(self, X):
		X,y = self.ensure(X)  
		outs = self(X)

		loss = self.criterion(outs.view(-1, outs.size(-1)),y.view(-1))
		return loss.item()

	@torch.no_grad()
	def generate(self, X, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
		X = self.ensure(X)

		for _ in range(max_new_tokens):
			X_running = X if X.size(1) <= self.context_size else X[:, -self.context_size:]
			
			logits = self(X_running)
			logits = logits[:, -1, :] / temperature
			
			# optionally crop the logits to only the top k options
			if top_k is not None:
				v, _ = torch.topk(logits, top_k)
				logits[logits < v[:, [-1]]] = -float('Inf')
			# apply softmax to convert logits to (normalized) probabilities
			probs = nn.functional.softmax(logits, dim=-1)
			# either sample from the distribution or take the most likely element
			if do_sample:
				X_next = torch.multinomial(probs, num_samples=1)
			else:
				_, X_next = torch.topk(probs, k=1, dim=-1)
			# append sampled index to the running sequence and continue
			X = torch.cat((X, X_next), dim=1)

		return X


	def configure_optimizer(self,model, optimizer_params):
		print(model)
		params = model.named_parameters()
		param_dict = {pn: p for pn, p in params}
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

		decay_params_n = [n for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params_n = [n for n, p in param_dict.items() if p.dim() < 2]

		optim_groups = [
			{"params": decay_params, "weight_decay": self.optimizer_params["optimizer_args"]["weight_decay"]},
			{"params": nodecay_params, "weight_decay": 0.0},
		]
		return optim_groups, optimizer_params

	def build_transformer(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = self.embed_dim*4, activation = nn.GELU(), batch_first= True, norm_first = True, dropout=self.dropout)

		arch_params = {
			"blocks":[nn.Embedding, OrderedPositionalEmbedding, nn.Dropout,MaskedTransformerEncoder, nn.LayerNorm, nn.Linear],
			"block_args":[
				{
					"num_embeddings": self.vocab_size,
					"embedding_dim" : self.embed_dim,
				},
				{
					"num_embeddings" : self.context_size,
					"embedding_dim" : self.embed_dim
				},
				{
					"p":self.dropout
				},
				{
					"encoder_layer":encoder,
					"num_layers":self.num_layers,
				},
				{
					"normalized_shape" : self.embed_dim
				},
				{
					"in_features" : self.embed_dim,
					"out_features":self.vocab_size,
					"bias":False
				}
			],
		}

		self.optimizer_params["configure_optimizer"] = self.configure_optimizer
		return {
			"arch_params":arch_params,
			"optimizer_params":self.optimizer_params,
			"torch_compile" : self.torch_compile,
		}