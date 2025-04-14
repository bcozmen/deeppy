
import torch
import torch.nn as nn

from deeppy.models.base_model import BaseModel
from deeppy.models.network import Network

class B_Vae(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]
	optimize_return_labels = ["Loss", "MSE Loss", "KL Loss"]
	def __init__(self, network_params,  beta,  device = None, criterion = nn.MSELoss()):

		super().__init__(device= device, criterion = criterion)
		self.beta = beta
		self.network_params = network_params

		self.net = Network(**network_params).to(self.device)
		
		self.params = [network_params,  beta,  device, criterion ]
		self.nets = [self.net]
		self.train()
	
	def init_objects(self):
		pass	

	def predict(self, X):
		X = self.ensure(X)
		latent = self.net.encode(X)
		z, mu, logvar = self.reparametrize(latent)
		y_pred = self.net.decode(z)

		return y_pred, mu, logvar

	
	def optimize(self, X):
		X,y = self.ensure(X)
		y_pred, mu, logvar = self.predict(X)
		con_loss = self.criterion(y_pred, y)  
		kl_loss = self.kl_loss(mu, logvar)

		loss = con_loss + self.beta * kl_loss
		self.net.back_propagate(loss)

		return loss.item(), con_loss.item(), kl_loss.item()

	def test(self, X):
		X,y = self.ensure(X)
		
		with torch.no_grad():
			y_pred, mu, logvar = self.predict(X)
		
			con_loss = self.criterion(y_pred, y)  
			kl_loss = self.beta * self.kl_loss(mu, logvar)
			loss = con_loss + kl_loss

		return loss.item(), con_loss.item(), kl_loss.item()
	

	def kl_loss(self, mu, logvar):
		kl = 0.5 * (torch.pow(mu,2) + torch.exp(logvar) - logvar - 1)
		return kl.sum(1).mean(0)

	def reparametrize(self,latent):
		latent_size = latent.shape[1]//2
		mu, logvar = latent[:, :latent_size], latent[:, latent_size:]
		std = torch.exp(0.5 *logvar)
		eps = torch.randn_like(std, device = self.device, dtype = torch.float32)

		z = mu + eps * std

		return z, mu, logvar

	
