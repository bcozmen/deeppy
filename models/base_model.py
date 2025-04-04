import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel():
	def __init__(self, device, criterion):
		self.device = device
		self.criterion = criterion