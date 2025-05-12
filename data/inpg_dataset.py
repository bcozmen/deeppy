from deeppy.data.base import Base


from torch.utils.data import Dataset, DataLoader, random_split
import torch


class IngpData(Dataset):
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
        if window_size is None:
            self.window_size = self.get_max_token_size(self.aligned_model)
    
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

    def tokenize_model(self,model):
        pass
    def get_max_token_size(self):
        N = self.extract_hash_encoding_structure(self.aligned_model)
        W = self.extract_mlp_weights(self.aligned_model)
        max_token = max(max_token,N["global_info"]["embedding_dim"])
        for key in W.keys():
            for l in W[key].keys():
                s = torch.prod(torch.tensor(W[key][l]["shape"][1:]))
                max_token = max(max_token,s)
        return max_token
    def extract_hash_encoding_structure(model_weights):
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
        num_levels = self.config["num_levels"]
        level_dim = self.config["level_dim"]
        input_dim = self.config["input_dim"]
        log2_hashmap_size = self.config["log2_hashmap_size"]
        base_resolution = self.config["base_resolution"]
        embeddings = model_weights['_orig_mod.grid_encoder.embeddings']
        
        # Calculate per-level parameters
        max_params = 2 ** log2_hashmap_size
        per_level_scale = np.exp2(np.log2(2048 / base_resolution) / (num_levels - 1))
        
        # Initialize structure to store weights
        hash_structure = {}
        offset = 0
        
        for level in range(num_levels):
            # Calculate resolution at this level
            resolution = int(np.ceil(base_resolution * (per_level_scale ** level)))
            
            # Calculate number of parameters for this level
            params_in_level = min(max_params, (resolution) ** input_dim)
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible by 8
            
            # Extract weights for this level
            level_weights = embeddings[offset:offset + params_in_level]
            
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
        
        return hash_structure

    def extract_mlp_weights(model_weights):
        """Extract geometric and view-dependent MLP weights from the model."""
        geometry_layers = {}
        view_mlp_layers = {}
        
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


