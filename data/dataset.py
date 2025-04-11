from deeppy.data.base import Base


from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle
import warnings


class FromLoader(Base):
    def __init__(self, train_loader, test_loader = [], valid_loader = [],
                batch_size = 64):
        super().__init__(self, batch_size = batch_size)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
    def save(self,file_name):
        warnings.warn("Save for train loader is not possible. Please save your dataset")

    def load(self, file_name):
        pass

class DataGetter(Base):
    def __init__(self,X=None, y= None, X_valid = None, 
                 test_size = 0.2, test_batch_size = None, valid_batch_size = None,
                 batch_size = 64, num_workers = 0, pin_memory=True, shuffle = True):
        super().__init__(batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = shuffle)

        if test_size < 0 or test_size > 1:
            raise ValueError("Invalid test size")

        self.test_size = test_size
        
        self.test_batch_size = test_batch_size
        self.valid_batch_size = valid_batch_size

        self.dataset = Data(X=X,y=y)
        self.valid = Data(X=X_valid)

        if X is not None:
            self.create_loaders()

    
    def save(self,file_name):
        params = {
            "train_dataset" : self.train_dataset,
            "test_dataset" : self.test_dataset,
            "valid"        : self.valid,
            "dataset"     : self.dataset
        }
        torch.save(params, file_name + '/memory.pkl')

    def load(self, file_name):
        params = torch.load(file_name  + '/memory.pkl', weights_only = False)
        self.train_dataset = params["train_dataset"]
        self.test_dataset = params["test_dataset"]
        self.valid = params["valid"] 
        self.dataset = params["dataset"]

    def train_data(self):
        return next(iter(self.train_loader))
    def test_data(self):
        return next(iter(self.test_loader))
    def valid_data(self):
        return next(iter(self.valid_loader))

    def create_loaders(self):
        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size

        if self.valid_batch_size is None:
            self.valid_batch_size = self.batch_size

        test_size = int(self.test_size * len(self.dataset))
        train_size = len(self.dataset) - test_size

        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=self.num_workers)
        if test_size > 0:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=self.num_workers)

        if self.valid.X is not None:
            self.valid_loader = DataLoader(self.valid, batch_size=self.valid_batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

class Data(Dataset):
    # (Batch_size, dimentions)
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)  # Total number of samples

    def __add__(self):
        pass

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx],
        return self.X[idx], self.y[idx]





