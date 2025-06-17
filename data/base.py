from abc import ABC, abstractmethod
from deeppy.utils import print_args
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset


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
        return len(self.train_dataset) + len(self.test_dataset) + len(self.valid_dataset)

    def train_data(self):
        try:
            X = next(self.train_iter)
        except:
            train_iter = iter(self.train_loader)
            self.train_iter = train_iter
            X = next(train_iter)
        return tuple(X)

    def test_data(self):
        try:
            X = next(self.test_iter)
        except:
            test_iter = iter(self.test_loader)
            self.test_iter = test_iter
            X = next(test_iter)
        return tuple(X)

    def valid_data(self):
        try:
            X = next(self.valid_iter)
        except:
            valid_iter = iter(self.valid_loader)
            self.valid_iter = valid_iter
            X = next(valid_iter)
        return tuple(X)


    @abstractmethod
    def save(self,file_name):
        pass

    @abstractmethod
    def load(self, file_name):
        pass


class DatasetLoader(Base):
    def __init__(self, data, splits = [0.8, 0.1, 0.1], file_name = None,
                batch_size = 64, num_workers = 0, pin_memory=True, shuffle = True):
        super().__init__(batch_size = batch_size,  num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
        self.data = data
        if file_name is not None:
            self.load(data, file_name)
        else:
            if not torch.is_tensor(splits):
                splits = torch.tensor(splits)
            if splits.sum() != 1:
                raise ValueError("Splits must sum to 1.0")
            
            self.splits = splits
            
            self.prepare(data)
        
    def prepare(self, data):
        total_length = len(data)
        lengths = torch.floor(torch.tensor(self.splits) * total_length).to(torch.int64)
        lengths[0] += total_length - lengths.sum()

        self.train_dataset, self.test_dataset,self.valid_dataset = random_split(data, lengths.tolist())

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))
        if len(self.test_dataset) > 0:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))
        if len(self.valid_dataset) > 0:
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))

    def save(self,file_name):
        split_indices = {
            "train": self.train_dataset.indices,
            "valid": self.valid_dataset.indices,
            "test": self.test_dataset.indices
        }
        torch.save(split_indices, file_name + '/split_indices.pkl')

    def load(self, data, file_name):
        split_indices = torch.load(file_name + '/split_indices.pkl', weights_only = False)
        self.train_dataset = Subset(data, split_indices["train"])
        self.valid_dataset = Subset(data, split_indices["valid"])
        self.test_dataset = Subset(data, split_indices["test"])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=min(self.num_workers,self.batch_size))



