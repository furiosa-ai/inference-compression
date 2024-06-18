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
    
    