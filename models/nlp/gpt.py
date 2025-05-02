import torch
import torch.nn as nn



from deeppy.utils import print_args
from deeppy.models.network import Network
from deeppy.models.base_model import BaseModel

class GPTPositionalEncoder(nn.Module):
	print_args = classmethod(print_args)
	dependencies = []
	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		self.embed = nn.Embedding(num_embeddings,embedding_dim)

	def forward(self,x):
		t = x.shape[1]
		pos = torch.arange(0, t, dtype=torch.long, device = x.device).unsqueeze(0) 
		return x + self.embed(pos)

class GPT(BaseModel):
	dependencies = []
	optimize_return_labels = ["Loss"]
	def __init__(self, optimizer_params, vocab_size = 3, embed_dim=48, num_heads=3, num_layers=3, max_seq_len=11, device = None, criterion = nn.CrossEntropyLoss()):
		super().__init__(device = device, criterion=criterion)

		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.max_seq_len = max_seq_len
		self.num_heads = num_heads
		self.num_layers = num_layers


		encoder = nn.TransformerEncoderLayer(d_model = embed_dim, nhead= num_heads, dim_feedforward = embed_dim*4, activation = nn.GELU(), batch_first= True, norm_first = True)

		arch_params = {
			"blocks":[nn.Embedding, GPTPositionalEncoder, nn.Dropout,nn.TransformerEncoder, nn.LayerNorm, nn.Linear],
			"block_args":[
				{
					"num_embeddings": vocab_size,
					"embedding_dim" : embed_dim,
				},
				{
					"num_embeddings" : max_seq_len,
					"embedding_dim" : embed_dim
				},
				{
					"p":0.1
				},
				{
					"encoder_layer":encoder,
					"num_layers":num_layers,
				},
				{
					"normalized_shape" : embed_dim
				},
				{
					"in_features" : embed_dim,
					"out_features":vocab_size,
					"bias":False
				}
			],
		}

		network_params = {
			"arch_params":arch_params,
			"optimizer_params":optimizer_params,
		}

		self.net = Network(**network_params).to(self.device)
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

		y0 = y[0]
		ysize = len(y0[y0!=-1])


		y = y[:,-ysize:]
		X = self.generate(X, max_new_tokens = ysize)[:,-ysize:]

		r = (X == y).flatten().cpu()
		return r.sum() / len(r)

	@torch.no_grad()
	def generate(self, X, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
		X = self.ensure(X)
		for _ in range(max_new_tokens):
			X = X if X.size(1) <= self.max_seq_len else X[:, -self.max_seq_len:]
			
			logits = self(X)
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