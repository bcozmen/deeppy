from networks.network import Network
import torch
import torch.nn as nn
import torch.optim as optim


class B_Vae():
	def __init__(self, network_params, latent_size, beta,  device = None, criterion = nn.MSELoss):
		if device is None:
		    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
		    self.device = device
		self.beta = beta
		
		self.latent_size = latent_size
		self.criterion = criterion()

		network_params["device"] = device

		self.model = Network(**network_params)
		self.train()
	def train(self):
		self.model.train()
		self.training = True
	def eval(self):
		self.model.eval()
		self.training = False

	def predict(self, X):
		X= self.model.ensure_tensor_device(X)
		latent = self.model.encode(X)
		z, mu, logvar = self.reparametrize(latent)
		y_pred = self.model.decode(z)

		return y_pred, mu, logvar
	
	def optimize(self, *X):
		X,y = map(self.model.ensure_tensor_device,list(X))
		y_pred, mu, logvar = self.predict(X)
		con_loss = self.criterion(y_pred, y)  
		kl_loss = self.beta * self.kl_loss(mu, logvar)

		loss = con_loss + kl_loss
		self.back_propagate(loss)

		return loss.item(), con_loss.item(), kl_loss.item()

	def test(self, *X):
		X,y = map(self.model.ensure_tensor_device,list(X))
		
		with torch.no_grad():
			y_pred, mu, logvar = self.predict(X)
		
			con_loss = self.criterion(y_pred, y)  
			kl_loss = self.beta * self.kl_loss(mu, logvar)
			loss = con_loss + kl_loss

		return loss.item(), con_loss.item(), kl_loss.item()
	def reparametrize(self,latent):
		mu, logvar = latent[:, self.latent_size:], latent[:, :self.latent_size]

		std = torch.exp(logvar / 2)
		eps = torch.randn_like(std, device = self.device, dtype = torch.float32)

		z = mu + eps * std

		return z, mu, logvar

	def kl_loss(self, mu, logvar):
		kl = 0.5 * (torch.pow(mu,2) + torch.exp(logvar) - logvar - 1)
		return kl.sum(1).mean(0)

	def back_propagate(self, loss):
		self.model.back_propagate(loss)