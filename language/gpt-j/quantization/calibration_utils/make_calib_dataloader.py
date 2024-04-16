import torch
from dataset import Dataset 
from torch.utils.data import DataLoader

__all__ = ["make_calib_dataloader"]


def make_calib_dataloader(calib_dataset_path, batch_size):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(
                len(data_object.source_encoded_input_ids[idx][0]))})
    
    return DataLoader(data_list, batch_size)