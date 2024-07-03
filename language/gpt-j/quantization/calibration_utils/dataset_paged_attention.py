from dataset import Dataset
import torch

class Dataset_for_paged_attention(Dataset):
    def __init__ (self, data_list, total_block_space, input_metadata = None):
        self.data_list = data_list
        self.total_block_space = total_block_space
        self.input_metadata = input_metadata 
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        input_kwargs = self.data_list[idx]
        input_kwargs["past_key_values"] = self.total_block_space
        if self.input_metadata:
            input_kwargs["input_metadata"] = self.input_metadata
        
        return input_kwargs
    
    def move_common_tensors_to_device(self, device, device_map = None):
        if self.input_metadata:
            for idx in range(len(self.input_metadata)):
                if not isinstance(self.input_metadata[idx], torch.Tensor):
                    continue
                self.input_metadata[idx] = self.input_metadata[idx].to(device)
        
        for type_idx in range(len(self.total_block_space)):
            self.total_block_space[type_idx] = list(self.total_block_space[type_idx])
            for block_idx in range(len(self.total_block_space[type_idx])):
                kv_cache_device = device
                if device_map is not None and len(device_map) > 0:
                    if isinstance(device[str(type_idx)], int):
                        kv_cache_device = device[str(type_idx)]
                self.total_block_space[type_idx][block_idx] = self.total_block_space[type_idx][block_idx].to(kv_cache_device)
                            
                
    def init_total_block_space(self, device, device_map = None):
        tensor_shape = self.total_block_space[0][0].shape
        dtype = self.total_block_space[0][0].dtype
        for type_idx in range(len(self.total_block_space)):
            for block_idx in range(len(self.total_block_space[type_idx])):
                kv_cache_device = device
                if device_map is not None and len(device_map) > 0:
                    if isinstance(device[str(type_idx)], int):
                        kv_cache_device = device[str(type_idx)]
                self.total_block_space[type_idx][block_idx] = torch.zeros(tensor_shape, dtype = dtype).to(kv_cache_device)
        
