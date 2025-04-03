from networks.network import Network
import torch
import torch.nn as nn
import torch.optim as optim


class B_Vae():
	def __init__(self, network_params, latent_size,  device = None, criterion = nn.MSELoss,
				beta):
		if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
		self.beta = beta
		
		self.latent_size = latent_size
		self.criterion = criterion()

		network_params["device"] = device

		self.model = Network(**network_params)

	def optimize(self, *X):
		X,y = X

		latent = self.model.encode(X)
		z, mu, logvar = self.reparametrize(latent)
		y_pred = self.model.decode(z)

		loss = self.criterion(y_pred, y) + self.beta * self.kl_loss(mu, logvar)
		self.back_propagate(loss)

	def reparametrize(self,latent):
		mu, logvar = latent[:, self.latent_size:], latent[:, :self.latent_size]

		std = torch.exp(logvar / 2)
		eps = torch.randn_like(std)

		z = mu + eps * std

		return z, mu, logvar

	def kl_loss(self, mu, logvar):
		kl = -0.5 * (torch.pow(mu,2) - torch.exp(logvar) + logvar + 1)
		return kl.sum(1).mean(0)

	def back_propagate(self, loss):
		self.model.back_propagate(loss)