import torch
import torch.nn as nn
import torch.optim as optim

class Scheduler():
	def __init__(self, optimizer, scheduler, gamma = 0.9):
		self.scheduler = scheduler(optimizer, gamma = gamma)
	def step(self):
		self.scheduler.step()