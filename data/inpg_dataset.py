#LOAD HALF PRECISION ALWAYS

from deeppy.data.base import Base


from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import glob
import json
import numpy as np
import h5py


class IngpData(Dataset):
    def __init__(self, data_path, config, window_size = None, token_size = None, max_layer_width = 64,  augment = False):
        self.data_path = data_path
        self.config = config
        self.augment = augment

        

        self.window_size = window_size
        self.hash_chunk_size = self.window_size - 53 # 53 is the size of the MLP tokens
        self.token_size = token_size
        self.max_layer_width = max_layer_width

        if self.token_size % self.max_layer_width != 0:
            raise ValueError("Invalid token size or max_layer_width")
        
        self.load_object_paths()
        
        self.max_positions = self[0][1].max(axis=0).values + 1
        self.max_positions[2] = (self.hash_table_indices_end[-1]  - self.hash_table_indices_end[-2] ).item()


    def __len__(self):
        if self.augment:
            return len(self.all_objects)
        else:
            return len(self.all_objects_2d)
    def __add__(self):
        pass

    def __getitem__(self, idx):
        #Random index
        #Sample (window_size - hash_chunk_size )points in 3D space (512,3)

        #Turn them into hash indices (16 , 512, 8)
        # (16, 512*8)
        if self.augment:
            rand_ix = torch.randint(low=0, high=self.hash_table_indices_end[-1].item() - (self.hash_chunk_size), size=(1,)).item()
            indices = torch.arange(rand_ix, rand_ix + self.hash_chunk_size)
        else:
            raise ValueError("Not implemented for non-augment mode")

        if self.augment:
            path, transform = self.all_objects[idx]

            t1,p1,m1,a1 = self.get_object_sample(path, indices)
            r1 = transform["scene_info"]["current_rotation"]
            r1 = torch.deg2rad(torch.tensor(r1))

            #check if there is any none in returns
            if torch.any(torch.isnan(t1)) or torch.any(torch.isnan(p1)) or torch.any(torch.isnan(m1)) or torch.any(torch.isnan(r1)):
                print(f"NaN values found in object {path} at index {idx}")
                return None, None, None, None, None, None, None, None
            return t1, p1, m1, r1, a1, torch.clone(p1), torch.clone(m1), torch.clone(r1)
        else:
            object_parent_path = self.all_objects_2d[idx]
            idx_child = torch.randperm(len(object_parent_path))[:2] 
            obj1_path, obj_1_transform = object_parent_path[idx_child[0]]
            obj2_path, obj_2_transform = object_parent_path[idx_child[1]]

            t1,p1,m1,_ = self.get_object_sample(obj1, indices)
            r1 = obj_1_transform["scene_info"]["current_rotation"]
            r1 = torch.deg2rad(torch.tensor(r1))

            t2,p2,m2,_ = self.get_object_sample(obj2, indices)
            r2 = obj_2_transform["scene_info"]["current_rotation"]
            r2 = torch.deg2rad(torch.tensor(r2))

            #check if there is any none in returns
            if torch.any(torch.isnan(t1)) or torch.any(torch.isnan(p1)) or torch.any(torch.isnan(m1)) or torch.any(torch.isnan(r1)) or \
               torch.any(torch.isnan(t2)) or torch.any(torch.isnan(p2)) or torch.any(torch.isnan(m2)) or torch.any(torch.isnan(r2)):
                print(f"NaN values found in object {obj1_path} or {obj2_path} at index {idx}")
                return None, None, None, None, None, None, None, None
            return t1, p1, m1, r1, t2, p2, m2, r2

    def get_object_sample(self, obj_path, indices):
        

        #(hash_chunk_size, token_size)
        hash_t, hash_p, hash_m, hash_aug = self.load_hash_consecutive(obj_path, indices)
        
        #(window_size - hash_chunk_size, token_size)
        mlp, mlp_aug = self.load_mlp_weights(obj_path)        
        mlp_t, mlp_p, mlp_m = mlp
        mlp_p[:,0] = torch.arange(self.hash_table_indices_end[-1].item()+1, self.hash_table_indices_end[-1].item()+1 + len(mlp_p[:,2]))  # Set position indices for MLP tokens

        return torch.vstack([hash_t, mlp_t]), torch.vstack([hash_p, mlp_p]), torch.vstack([hash_m, mlp_m]), torch.vstack([hash_aug, mlp_aug])
    

    def load_hash_consecutive(self, path, global_indices):
        layers = torch.searchsorted(self.hash_table_indices_end, global_indices, right=True)
        local_starts = self.hash_table_indices_start[layers]

        # Local index is just offset from start
        local_indices = (global_indices - local_starts) * (self.token_size // 2)

        pos = torch.stack([global_indices,layers, local_indices/ (self.token_size // 2)]).T.to(dtype=torch.int64)  # (num_points, 2)

        unique_layers = torch.unique(layers)
        tokens, masks = [], []

        for layer in unique_layers:
            mask = (layers == layer)
            indices = local_indices[mask]  # All sub-indices for this layer
            
            with h5py.File(path, 'r') as f:
                dset = f[f"level_{layer.item()}"]
                chunk = torch.from_numpy(dset[indices[0] : indices[-1] + (self.token_size // 2)])  # Load the entire chunk
            
            mask = torch.ones_like(chunk)
            n_layers_per_token = (self.token_size // 2) 
            pad_axis_0 = (n_layers_per_token - (chunk.shape[0] % n_layers_per_token)) % n_layers_per_token
            if pad_axis_0 > 0:
                chunk = torch.cat([chunk, torch.zeros(pad_axis_0, chunk.shape[1])], dim=0) 
                mask = torch.cat([mask, torch.zeros(pad_axis_0, mask.shape[1])], dim=0)
            
            tokens.append(chunk)
            masks.append(mask)
        tokens = torch.cat(tokens, dim=0)  # Concatenate all layers
        tokens = tokens.reshape(-1, self.token_size)  # Reshape to

        masks = torch.cat(masks, dim=0)  # Concatenate all masks
        masks = masks.reshape(-1, self.token_size)  # Reshape to match tokens


        return tokens, pos, masks, self.augment_layer(tokens)

    def load_hash_every_layer(self, path, indices):
        """
        indices : (num_layers, num_points * 8) Tensor / numpy array
        """
        #(16,512*8)
        return_data = []
        with h5py.File(path, 'r') as f:
            for i,ix in enumerate(indices):
                dset = f[f"level_{i}"] 
                #sort
                indices = np.sort(ix)
                chunk = torch.from_numpy(dset[indices]) # (512*8,2)
                chunk = chunk.reshape(-1,8,2) #(512,8,2)
                return_data.append(chunk)
        
        return_data = torch.stack(return_data) #(16,512,8,2)
        #turn it to (512,8,16,2) and then flatten to (512,256)
        token = return_data.permute(1,2,0,3).reshape(-1, self.token_size)
        augmented = None
        if self.augment:
            augmented = self.augment_layer(token)
        mask = torch.ones_like(token)
        #(1,256)
        pos = torch.randint(0,512,size = (token.shape[0],2))

        return token, pos, mask, augmented
    
    def load_hash_info(self):
        hash_table_indices = []
        with h5py.File(self.all_objects[0][0], 'r') as f:
            for i in range(16):
                hash_table_indices.append(f[f"level_{i}"].shape[0])
        hash_table_indices = torch.ceil(torch.tensor(hash_table_indices) / (self.token_size // 2))
        self.hash_table_indices_end = torch.cumsum(hash_table_indices, dim=0).to(dtype=torch.int)
        self.hash_table_indices_start = torch.cat([torch.tensor([0]), self.hash_table_indices_end[:-1]])
    def load_object_paths(self):
        
        self.objects = glob.glob(self.data_path + "/*")
        self.all_objects_2d = []
        self.all_objects = []
        for o in self.objects:
            objs_path = glob.glob(o + "/*")
            this_object = []
            for in_obj in objs_path:

                with open(in_obj + "/transforms.json", "r") as file:
                    transform = json.load(file)
                    del transform["frames"]
                data = (in_obj + "/checkpoints/final.h5",transform)
                this_object.append(data)
                self.all_objects.append(data)
            self.all_objects_2d.append(this_object)
        self.load_hash_info()
    

    def load_mlp_weights(self,path):
        """Load MLP weights from a file and tokenize them into layers.  
        Args:
            path (str): Path to the HDF5 file containing weights.
        Returns:
            list: A list of tokenized MLP layers, each represented as a tensor.
        """
        geometry_layers, view_layers = [], []
        augmented_geometry_layers, augmented_view_layers = [], []
        
        with h5py.File(path, 'r') as f:
            for i in range(3):
                dset1, dset2= f[f"geo_layer_{i}"], f[f"view_layer_{i}"]
                chunk1 = torch.from_numpy(dset1[:])
                chunk2 = torch.from_numpy(dset2[:])

                if self.augment:
                    augment1 = self.augment_layer(chunk1)
                    augment2 = self.augment_layer(chunk2)

                    augmented_geometry_layers.append(self.tokenize_mlp_layer(augment1, layer_ix= i + 16))
                    augmented_view_layers.append(self.tokenize_mlp_layer(augment2, layer_ix= i + 16 + 3))
                # Tokenize each layer
                geometry_layers.append(self.tokenize_mlp_layer(chunk1, layer_ix= i + 16))
                view_layers.append(self.tokenize_mlp_layer(chunk2, layer_ix= i + 16 + 3))

        #(6 , X , token_size) -> (1, 6X, token_size)
        layers = geometry_layers + view_layers
        layers = [torch.vstack(k) for k in zip(*layers)]

        if self.augment:
            augmented_layers = augmented_geometry_layers + augmented_view_layers
            augmented_layers = [torch.vstack(k) for k in zip(*augmented_layers)]
        return layers, augmented_layers[0]
        
    def tokenize_mlp_layer(self, w, layer_ix = None):
        pad = self.max_layer_width - w.shape[1]
        mask = torch.ones_like(w)
        
        # w - > (x , max_layer_width)
        if pad > 0:
            w = nn.functional.pad(w, (0, pad))
            mask = nn.functional.pad(mask, (0,pad))

        n_layers_per_token = (self.token_size // self.max_layer_width)
        pad_axis_0 = (n_layers_per_token - (w.shape[0] % n_layers_per_token)) % n_layers_per_token
        
        # (x, max_layer_width) -> (n_layers_per_token *k , max_layer_width)
        if pad_axis_0 > 0:
            w = torch.cat([w, torch.zeros(pad_axis_0, w.shape[1])], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_axis_0, mask.shape[1])], dim=0)

        #(Group each 4 rows into one row)
        w = w.view(-1, n_layers_per_token , self.max_layer_width).view(-1,self.token_size)
        mask = mask.view(-1, n_layers_per_token , self.max_layer_width).view(-1,self.token_size)
        
        pos = torch.zeros((w.shape[0], 3), dtype=torch.int64)
        pos[:,1] = torch.full_like(pos[:,0], layer_ix)
        pos[:,2] = torch.arange(w.shape[0])
        return w,pos,mask

    def augment_layer(self, w):
        """Apply data augmentation to a single MLP layer."""
        # Example augmentation: add noise
        noise = 0.01 * torch.randn_like(w)
        return w + noise

