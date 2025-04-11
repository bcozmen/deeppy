import torch
from deeppy.utils import print_args
class Epsilon():
    print_args = classmethod(print_args)
    def __init__(self, eps = 0.9, eps_end = 0.05, eps_decay =1000, random_generator = None):
        if eps is None:
            eps = 0
            eps_end = 0
        self.eps = eps
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = torch.tensor(0)

        self.random_generator = random_generator
    def __call__(self):
        return self.eps
    
    def update(self):
        self.steps_done += 1
        self.eps = self.eps_end + (self.eps - self.eps_end) * torch.exp(-1. * self.steps_done / self.eps_decay)


    def generate(self):
        return self.random_generator()

class TargetUpdater():
    def __init__(self, main, target, tau = 0.005):
        self.tau = tau
        self.main = main
        self.target = target

        self.target.load_state_dict(self.main.state_dict())
        
        for param in self.target.parameters():
            param.requires_grad = False

    def update(self):
        target_state_dict = self.target.state_dict()
        main_state_dict = self.main.state_dict()
        for key in main_state_dict:
            target_state_dict[key] = main_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_state_dict)