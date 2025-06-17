from abc import ABC, abstractmethod
from deeppy.utils import print_args
import torch

class Base(ABC):
    print_args = classmethod(print_args)
    def __init__(self, batch_size = 64, num_workers = 0 ,pin_memory=True, shuffle = True):
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.train_loader = []
        self.test_loader = []
        self.valid_loader = []
    
    def __len__(self):
        return len(self.train_loader) + len(self.test_loader)

    def train_data(self):
        try:
            X = next(self.train_iter)
        except:
            self.train_iter = iter(self.train_loader)
            X = next(self.train_iter)
        return tuple(X)

    def test_data(self):
        try:
            X = next(self.test_iter)
        except:
            self.test_iter = iter(self.test_loader)
            X = next(self.test_iter)
        return tuple(X)

    def valid_data(self):
        try:
            X = next(self.valid_iter)
        except:
            self.valid_iter = iter(self.valid_loader)
            X = next(self.valid_iter)
        return tuple(X)


    @abstractmethod
    def save(self,file_name):
        pass

    @abstractmethod
    def load(self, file_name):
        pass



