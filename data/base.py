from abc import ABC, abstractmethod
from deeppy.utils import print_args
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, Sampler


class DatasetBase(ABC):
    print_args = classmethod(print_args)
    def __init__(self, batch_size = 64, dataloader_args = {}):
        self.device = torch.device("cpu")

        self.dataloader_args = dataloader_args
        self.batch_size = batch_size
        
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


class UniquePerBatchSampler(Sampler):
    def __init__(self, dataset_size, num_repeats, batch_size):
        if batch_size > dataset_size:
            raise ValueError("Batch size cannot be greater than dataset size for uniqueness.")
        self.dataset_size = dataset_size
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        self.len_per_epoch = int(dataset_size / batch_size) * batch_size

    def __iter__(self):
        # Repeat indices num_repeats times, shuffle each repeat independently
        indices = torch.stack([
            torch.randperm(self.dataset_size) for _ in range(self.num_repeats)
        ])[:,:self.len_per_epoch].flatten()

        return iter(indices.tolist())

    def __len__(self):
        return (self.len_per_epoch * self.num_repeats) 

class DatasetLoader(DatasetBase):
    def __init__(self, data, test_data = None, splits = None, file_name = None, repeat = 1,
                batch_size = 64, dataloader_args = {}):
        super().__init__(batch_size = batch_size,  dataloader_args = dataloader_args)

        self.repeat = repeat
        if repeat > 1 :
            dataloader_args['shuffle'] = False
        

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

        if self.repeat > 1:
            self.dataloader_args['sampler'] = UniquePerBatchSampler(len(self.train_dataset), self.repeat, self.batch_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, **self.dataloader_args)
        
        
        if len(self.test_dataset) > 0:
            if self.repeat > 1:
                self.dataloader_args['sampler'] = UniquePerBatchSampler(len(self.test_dataset), self.repeat, self.batch_size)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_args)
        if len(self.valid_dataset) > 0:
            if self.repeat > 1:
                self.dataloader_args['sampler'] = UniquePerBatchSampler(len(self.valid_dataset), self.repeat, self.batch_size)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, **self.dataloader_args)

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

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, **self.dataloader_args)
        if len(self.test_dataset) > 0:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_args)
        if len(self.valid_dataset) > 0:
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, **self.dataloader_args)

class DatasetLoaderEasy(DatasetBase):
    def __init__(self, train_dataset, test_dataset = None, batch_size = 64, dataloader_args = {}):
        super().__init__(batch_size = batch_size,  dataloader_args = dataloader_args)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, **self.dataloader_args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_args)
    
    def save(self,file_name):
        pass

    def load(self, data, file_name):
        pass



