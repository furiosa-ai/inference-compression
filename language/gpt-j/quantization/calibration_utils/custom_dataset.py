from dataset import Dataset

class Dataset_with_total_block(Dataset):
    def __init__ (self, data_list, total_block_space):
        self.data_list = data_list
        self.total_block_space = total_block_space
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        input_kwargs = self.data_list[idx]
        input_kwargs["past_key_values"] = self.total_block_space
        
        return input_kwargs
    
    def move_total_block_to_device(self, device = None , device_map = None):
        if device == None and device_map == None:
            raise RuntimeError

        for type_idx in range(len(self.total_block_space)):
            self.total_block_space[type_idx] = list(self.total_block_space[type_idx])
            for block_idx in range(len(self.total_block_space[type_idx])):
                kv_cache_device = device
                if device_map is not None:
                    if isinstance(device[str(type_idx)], int):
                        kv_cache_device = device[str(type_idx)]
                self.total_block_space[type_idx][block_idx] = self.total_block_space[type_idx][block_idx].squeeze(0).to(kv_cache_device)
                            
                
        
        
    
    