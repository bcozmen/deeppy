from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer



class DataGetter():
    def __init__(self,X, batch_size = 64, test_size = 0.2, y= None, X_test = None, device = None, normalization = "uniform", task = "reg"):
        self.task = task
        self.dataset = Data(X=X,y=y,X_test = X_test, device = device, normalization = normalization, task = task)

        if self.task == "autoencoder":
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.dataset, batch_size=2048, shuffle=True)
class Data(Dataset):
    def __init__(self, X, y = None, X_test = None, device = None, normalization = "uniform", task = "reg"):
        self.normalization = normalization
        self.X = X
        self.y = y
        self.X_test = X_test
        self.device = device
        self.task = task
        
        self.create_scaler()
        self.normalize_data()
        self.load_to_device()


    def __len__(self):
        return len(self.X)  # Total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def transform(self, x):
        return torch.tensor(self.scaler.transform(x) ,dtype = torch.float32)

    def create_scaler(self):
        if self.normalization == "normal":
            self.scaler = StandardScaler()
        elif self.normalization == "uniform":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = FunctionTransformer(lambda x : x)
        if self.X_test is None:
            self.features = self.X
        else:
            self.features = torch.concat([X, X_test])
        self.scaler.fit(self.features)

    def normalize_data(self):
        if self.task == "autoencoder":
            self.X = self.transform(self.features)
            self.y = self.X
            
        else:
            self.X = self.transform(self.X)
            self.y = self.y
        if not X_test is None:
            self.X_test = self.transform(X_test)

    def load_to_device(self):
        if not self.device is None:
            self.X = self.X.to(device)

            if not X_test is None:
                self.X_test = self.X_test.to(device)
            self.y = self.y.to(device) 




