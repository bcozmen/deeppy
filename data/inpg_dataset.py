#LOAD HALF PRECISION ALWAYS

from deeppy.data.base import Base


from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import glob
import json

class PtToHDF5():
    def __init__(self):
        pass

class IngpData(Dataset):
    def __init__(self, data_path, config, window_size = None, hash_chunk_size, token_size = None, max_layer_width = 64):
        self.data_path = data_path
        self.config = config

        if self.token_size % self.max_layer_width != 0:
            raise ValueError("Invalid token size or max_layer_width")

        self.window_size = window_size
        self.hash_chunk_size = hash_chunk_size
        self.token_size = token_size
        self.max_layer_width = max_layer_width
        
        self.load_object_paths()

    def __len__(self):
        return len(self.all_objects)
    def __add__(self):
        pass

    def __getitem__(self, idx):
        #Random index
        file_path, transform = self.all_objects[idx]

        obj1, obj2 = 1 #

        t1,p1,m1 = self.get_object_sample(obj1)
        t2,p2,m2 = self.get_object_sample(obj2)

        start_chunk = torch.rand()
        return t1, p1, m1, t2, p2, m2, r1, r2

    def get_object_sample(self, obj_path):
        #Sample (window_size - hash_chunk_size )points in 3D space (512,3)

        #Turn them into hash indices (16 , 512, 8)
        # (16, 512*8)

        #(hash_chunk_size, token_size)
        hash_t, hash_p, hash_m = self.load_hash_chunk(obj_path)
        
        #(window_size - hash_chunk_size, token_size)
        mlp_t, mlp_p, mlp_m = self.load_mlp_weights(obj_path)

        return torch.cat([hash_t, mlp_t]), torch.cat([hash_p, mlp_p]), torch.cat([hash_m, mlp_m])
    
    def load_hash_chunk(self, path, indices):
        """
        indices : (num_layers, num_points * 8) Tensor / numpy array
        """
        #(16,512*8)
        return_data = []
        with h5py.File(path, 'r') as f:
            for i,ix in enumerate(indices):
                dset = f[f"layer{i}"] 
                chunk = dset[indices] # (512*8,2)
                chunk = chunk.reshape(-1,8,2) #(512,8,2)
                return_data.append(chunk)
        
        return_data = torch.from_numpy(np.asarray(return_data)) #(16,512,8,2)
        #turn it to (512,8,16,2) and then flatten to (512,256)
        mask = torch.ones_like(return_data)
        #(1,256)
        return return_data, pos, mask

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
                data = (in_obj + "/checkpoints/final.pth",transform)
                this_object.append(data)
                self.all_objects.append(data)
            self.all_objects_2d.append(this_object)


    

    

    def load_mlp_weights(self,path):
        geometry_layers, view_layers = [], []
        
        with h5py.File(path, 'r') as f:
            for i in range(3):
                dset1, dset2= f["geometry_layers{i}"], f["view_layers{i}"]
                chunk1, chunk2 = dset1, dset2
                geometry_layers.append(self.tokenize_mlp_layer(chunk1))
                view_layers.append(self.tokenize_mlp_layer(chunk2))
        
        #(6 , X , token_size) -> (1, 6X, token_size)
        layers = geometry_layers + view_layers
    def tokenize_mlp_layer(self, w):
        pad = self.max_layer_width - w.shape[1]
        mask = torch.ones_like(w)
        
        # w - > (x , max_layer_width)
        if pad > 0:
            w = nn.function.pad(w, (0, self.max_layer_width - w.shape[1]))
            mask = nn.functional.pad(mask, (0,max_len - w.shape[1]))     
        
        n_layers_per_token = (self.token_size // self.max_layer_width)
        pad_axis_0 = n_layers_per_token - (w.shape[0] % n_layers_per_token)
        
        # (x, max_layer_width) -> (n_layers_per_token *k , max_layer_width)
        if pad_axis_0 > 0:
            #pad

        #(Group each 4 rows into one row)
        
        max_len = max(w.shape[1] for w in weights)

        for ix,w in enumerate(weights):
            mask = torch.ones_like(w)

            x,y = w.shape
            token = nn.functional.pad(w, (0, max_len - w.shape[1]))
            mask = nn.functional.pad(mask, (0,max_len - w.shape[1]))
            tokens.append(token)
            masks.append(mask)

            layer_ix = torch.full(size=(token.shape[0],1), fill_value=ix) 
            layer_pos = torch.arange(token.shape[0]).unsqueeze(1)
            positions.append(torch.cat((layer_ix,layer_pos), dim =1))

        tokens, positions, masks = torch.cat(tokens), torch.cat(positions), torch.cat(masks)
        positions =  torch.cat((torch.arange(positions.size(0)).unsqueeze(1),positions), dim=1)
        # Pad each tensor to match max_len in second dimension

        
        return tokens, positions, masks
            
class IngpData_old(Dataset):
    def __init__(self,data_path, config, window_size = None):
        

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
        (model, position, mask, augment, position, mask) = self.cut_model(self.get_augmented_weights(file_path))
        
        rot = transform["scene_info"]["current_rotation"]
        rot = torch.deg2rad(torch.tensor(rot))
        return model, position, mask, augment, position, mask, rot


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
                data = (in_obj + "/checkpoints/final.pth",transform)
                this_object.append(data)
                self.all_objects.append(data)
            self.all_objects_2d.append(this_object)

        self.select_aligned_model()
    
    def data_augmentation(self,model):
        augment = torch.clone(model)
        #augment = self.permute(augment)
        #model, augment, position = self.cut_model(model, augment)
        augment = self.add_noise(augment)
        return (model, position, mask, augment, position, mask)

    def cut_model(self,models):
        start = torch.randint(models[0].shape[0]-self.window_size, size = (1,)).item()
        return (m[start:start + self.window_size] for m in models)

    def create_mask(self,model):
        return mask
    
    def permute(self, model):
        return model
    
    def add_noise(self,model, mask):
        aug_model = model + 0.1 * torch.randn(model.shape)
        return aug_model * mask

    def tokenize_model(self,model):
        pass
    
    def get_augmented_weights(self,file_path):
        w_dict = torch.load(file_path, map_location="cpu")['model']
        w = self.get_mlp_weigts(w_dict)
        if any(torch.isnan(t).any().item() for t in w):
            print(file_path)

        t,p,m = self.tokenize_mlp_weights(w)
        t_aug = self.add_noise(t,m)

        return  (t,p,m, t_aug, torch.clone(p), torch.clone(m))
    def load_torch_weights(self,file_path):
        """Load model weights from a checkpoint file."""
        try:
            weights = torch.load(file_path, map_location="cpu")
            return weights['model']
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def select_aligned_model(self):
        file_path, transform = self.all_objects[0]
        self.aligned_model = self.load_torch_weights(file_path)
        weights = self.get_mlp_weigts(self.aligned_model)
        self.token_size = max(w.shape[1] for w in weights)
        tokens,positions,masks = self.tokenize_mlp_weights(weights)
        self.max_positions = torch.max(positions,dim=0).values + 1
    

    def tokenize_mlp_weights(self, weights):
        positions = []
        tokens = []
        masks = []
        
        
        max_len = max(w.shape[1] for w in weights)

        for ix,w in enumerate(weights):
            mask = torch.ones_like(w)

            x,y = w.shape
            token = nn.functional.pad(w, (0, max_len - w.shape[1]))
            mask = nn.functional.pad(mask, (0,max_len - w.shape[1]))
            tokens.append(token)
            masks.append(mask)

            layer_ix = torch.full(size=(token.shape[0],1), fill_value=ix) 
            layer_pos = torch.arange(token.shape[0]).unsqueeze(1)
            positions.append(torch.cat((layer_ix,layer_pos), dim =1))

        tokens, positions, masks = torch.cat(tokens), torch.cat(positions), torch.cat(masks)
        positions =  torch.cat((torch.arange(positions.size(0)).unsqueeze(1),positions), dim=1)
        # Pad each tensor to match max_len in second dimension

        
        return tokens, positions, masks

    def get_mlp_weigts(self,model_weights):
        weights = []
        for i in range(self.config['mlp']['num_layers']):
            weight_key = f'_orig_mod.grid_mlp.net.{i}.weight'
            bias_key = f'_orig_mod.grid_mlp.net.{i}.bias'

            if weight_key in model_weights:
                w = model_weights[weight_key]
                w = w.view(w.shape[0],-1)
                if bias_key in model_weights:
                    b = model_weights[bias_key]
                    w = torch.cat([w,b], dim=1)
                weights.append(w)
                
        # Extract view-dependent MLP weights
        for i in range(self.config['mlp']['num_layers']):
            weight_key = f'_orig_mod.view_mlp.net.{i}.weight'
            bias_key = f'_orig_mod.view_mlp.net.{i}.bias'
            
            if weight_key in model_weights:
                w = model_weights[weight_key]
                w = w.view(w.shape[0],-1)
                if bias_key in model_weights:
                    b = model_weights[bias_key]
                    w = torch.cat([w,b], dim=1)
                weights.append(w)
        return weights
    def extract_hash_encoding_structure(self,model_weights):
        """
        Extract and organize hash encoding weights into hierarchical structure.
        
        Args:
            model_weights (dict): The loaded model weights dictionary
            num_levels (int): Number of levels in hash encoding
            level_dim (int): Dimension of encoding at each level
            input_dim (int): Input dimension (typically 3 for 3D)
            log2_hashmap_size (int): Log2 of maximum hash table size
            base_resolution (int): Base resolution of the grid
            
        Returns:
            dict: Hierarchical structure of hash encoding weights
        """
        # Extract hash encoding embeddings
        config = self.config["hash_encoding"]
        num_levels = config["num_levels"]
        level_dim = config["level_dim"]
        input_dim = config["input_dim"]
        log2_hashmap_size = config["log2_hashmap_size"]
        base_resolution = config["base_resolution"]
        embeddings = model_weights['_orig_mod.grid_encoder.embeddings']
        
        # Calculate per-level parameters
        max_params = 2 ** log2_hashmap_size
        per_level_scale = np.exp2(np.log2(2048 / base_resolution) / (num_levels - 1))
        
        # Initialize structure to store weights
        hash_structure = {}
        offset = 0

        weights = []

        
        for level in range(num_levels):
            # Calculate resolution at this level
            resolution = int(np.ceil(base_resolution * (per_level_scale ** level)))
            
            # Calculate number of parameters for this level
            params_in_level = min(max_params, (resolution) ** input_dim)
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible by 8
            
            # Extract weights for this level
            level_weights = embeddings[offset:offset + params_in_level]
            weights.append(weights)
            # Store level information
            hash_structure[f'level_{level}'] = {
                'resolution': resolution,
                'num_params': params_in_level,
                'weights': level_weights,
                'weights_shape': level_weights.shape,
                'scale': per_level_scale ** level
            }
            
            offset += params_in_level
        
        # Add global information
        hash_structure['global_info'] = {
            'total_params': offset,
            'embedding_dim': level_dim,
            'base_resolution': base_resolution,
            'max_resolution': int(np.ceil(base_resolution * (per_level_scale ** (num_levels-1)))),
            'per_level_scale': per_level_scale
        }
        
        return weights

    

    def extract_mlp_weightsa(self, model_weights):
        """Extract geometric and view-dependent MLP weights from the model."""
        geometry_layers = {}
        view_mlp_layers = {}

        weights = []
        
        # Extract geometry MLP weights
        for i in range(self.config['mlp']['num_layers']):
            weight_key = f'_orig_mod.grid_mlp.net.{i}.weight'
            bias_key = f'_orig_mod.grid_mlp.net.{i}.bias'
            
            if weight_key in model_weights:
                geometry_layers[f'layer_{i}'] = {
                    'weights': model_weights[weight_key],
                    'shape': model_weights[weight_key].shape
                }
                
                if bias_key in model_weights:
                    geometry_layers[f'layer_{i}']['bias'] = model_weights[bias_key]
        
        # Extract view-dependent MLP weights
        for i in range(self.config['mlp']['num_layers']):
            weight_key = f'_orig_mod.view_mlp.net.{i}.weight'
            bias_key = f'_orig_mod.view_mlp.net.{i}.bias'
            
            if weight_key in model_weights:
                view_mlp_layers[f'layer_{i}'] = {
                    'weights': model_weights[weight_key],
                    'shape': model_weights[weight_key].shape
                }
                
                if bias_key in model_weights:
                    view_mlp_layers[f'layer_{i}']['bias'] = model_weights[bias_key]
        
        return {
            'geometry_mlp': geometry_layers,
            'view_mlp': view_mlp_layers
        }


