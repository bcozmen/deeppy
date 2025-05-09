from deeppy.data.base import Base


from torch.utils.data import Dataset, DataLoader, random_split
import torch

class GPTText(Base):
    def __init__(self,train, tokenizer, context_size, test = None, valid = None,  batch_size = 64, test_size = 0.1):
        super().__init__(batch_size=batch_size)
        self.context_size = context_size
        self.tokenizer = tokenizer

        self.train_dataset = self.tokenizer.encode(train)

        if test is not None:
            self.test_data = self.tokenizer.encode(test)
        else:
            if test_size is not None:
            
                self.train_size = int((1-test_size) * len(self.train_dataset))
                self.test_dataset =  self.train_dataset[self.train_size:]
                self.train_dataset = self.train_dataset[:self.train_size]

        if valid is not None:
            self.valid_dataset = self.tokenizer.encode(valid)
        
    def train_data(self):
        ix = torch.randint(len(self.train_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.train_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.train_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y
    def test_data(self):
        ix = torch.randint(len(self.test_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.test_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.test_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y

    def valid_data(self):
        ix = torch.randint(len(self.valid_dataset) - self.context_size, (self.batch_size,))
        x = torch.stack([torch.tensor(self.valid_dataset[i: i+self.context_size]) for i in ix])
        y = torch.stack([torch.tensor(self.valid_dataset[i+1 : i+1+self.context_size]) for i in ix])

        return x,y

    def load(self):
        print("Save for GPTText is not possible. Please save your dataset")
    def save(self):
        print("Save for GPTText is not possible. Please save your dataset")

class FromLoader(Base):
    def __init__(self, train_loader, test_loader = [], valid_loader = [],
                batch_size = 64):
        super().__init__(batch_size = batch_size)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
    def train_data(self):
        return tuple(next(iter(self.train_loader)))
    def test_data(self):
        return tuple(next(iter(self.test_loader)))
    def valid_data(self):
        return tuple(next(iter(self.valid_loader)))
    def save(self,file_name):
        print("Save for train loader is not possible. Please save your dataset")
    def load(self, file_name):
        print("Save for train loader is not possible. Please save your dataset")

class DataGetter(Base):
    def __init__(self,X=None, y= None, X_valid = None, X_test = None, y_test = None, 
                 test_size = 0.2, test_batch_size = None, valid_batch_size = None,
                 batch_size = 64, num_workers = 0, pin_memory=True, shuffle = True):
        super().__init__(batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = shuffle)

        if test_size < 0 or test_size > 1:
            raise ValueError("Invalid test size")

        self.test_size = test_size
        
        self.test_batch_size = test_batch_size
        self.valid_batch_size = valid_batch_size

        self.train_dataset = Data(X=X,y=y)
        if X_test is not None:
            self.test_dataset = Data(X=X_test, y= y_test)
        else:
            self.test_dataset = None
            
        self.valid_dataset = Data(X=X_valid)

        if X is not None:
            self.create_loaders()

    
    def save(self,file_name):
        params = {
            "train_loader" : self.train_loader,
            "test_loader" : self.test_loader,
            "valid_loader"        : self.valid_loader,
        }
        torch.save(params, file_name + '/memory.pkl')

    def load(self, file_name):
        params = torch.load(file_name  + '/memory.pkl', weights_only = False)
        self.train_loader = params["train_loader"]
        self.test_loader = params["test_loader"]
        self.valid_loader = params["valid_loader"] 

    def train_data(self):
        return tuple(next(iter(self.train_loader)))
    def test_data(self):
        return tuple(next(iter(self.test_loader)))
    def valid_data(self):
        return tuple(next(iter(self.valid_loader)))

    def create_loaders(self):
        if self.test_batch_size is None:
            self.test_batch_size = self.batch_size

        if self.valid_batch_size is None:
            self.valid_batch_size = self.batch_size

        if self.test_dataset is None:
            self.test_size = int(self.test_size * len(self.train_dataset))
            train_size = len(self.train_dataset) - self.test_size

            self.train_dataset, self.test_dataset = random_split(self.train_dataset, [train_size, self.test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=self.num_workers)
        self.test_loader = None
        self.valid_loader = None
        if self.test_size > 0:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=self.shuffle, pin_memory=self.pin_memory, num_workers=self.num_workers)

        if self.valid_dataset.X is not None:
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



class InpgDataset(Dataset):
    def __init__(self,data_path, config, window_size):
        import glob
        import json

        self.data_path = data_path
        self.config = config
        self.window_size = window_size
        self.load_object_paths()
        
        
    def __len__(self):
        return len(self.all_objects)

    def __add__(self):
        pass

    def __getitem__(self, idx):
        file_path, transform = self.all_objects[idx]
        model = self.load_torch_weights(file_path)
        return self.data_augmentation(model)


    def load_object_paths(self):
        self.objects = glob.glob(self.data_path + "/*")
        self.all_objects_2d = []
        self.all_objects
        for o in self.objects:
            objs_path = glob.glob(o + "/*")
            this_object = []
            for in_obj in objs_path:
                with open(in_obj + "/transforms.json", "r") as file:
                    transform = json.load(file)
                    del transform["frames"]
                data = (in_obj + "/checkpoints/final.pth",transform)
                this_object.append(data)
                self.all_objects.append(data)
            self.all_objects_2d.append(this_object)

        self.select_aligned_model()
    
    def load_torch_weights(self,file_path):
        """Load model weights from a checkpoint file."""
        try:
            weights = torch.load(file_path, map_location=self.device)
            return weights['model']
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def select_aligned_model(self):
        file_path, transform = self.all_objects[0]
        self.aligned_model = self.load_torch_weights(file_path)
    
    def data_augmentation(self,model):
        augment = self.permute(model)
        model, augment, position = self.cut_model(model, augment)
        augment = self.add_noise(augment)
        mask = self.create_mask(model)
        return (model, position, mask, augment, position, mask)

    def cut_model(self,model, augment):
        return model,augment, 0

    def create_mask(self,model):
        return mask
    
    def permute(self, model):
        return model
    
    def add_noise(self,model):
        return model