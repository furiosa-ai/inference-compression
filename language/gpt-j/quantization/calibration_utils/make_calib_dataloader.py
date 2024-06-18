import torch
from dataset import Dataset 
from torch.utils.data import DataLoader

__all__ = ["make_calib_dataloader"]


def make_calib_dataloader(calib_dataset_path, batch_size, calib_without_padding):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        if calib_without_padding:
            data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(
                    len(data_object.source_encoded_input_ids[idx][0]))})
            
        else:
            bucket_size=2048
            starting_input_ids = data_object.source_encoded_input_ids[idx]
            batch_size, starting_input_len = starting_input_ids.shape
            bucketized_input_ids = torch.zeros((batch_size, bucket_size), dtype=torch.int)
            bucketized_input_ids[:, -starting_input_len:] = starting_input_ids
            
            starting_attention_mask =  data_object.source_encoded_attn_masks[idx]
            bucketized_attention_mask = torch.zeros((batch_size, bucket_size), dtype=torch.int)
            bucketized_attention_mask[:, -starting_input_len:] = starting_attention_mask
            
            starting_position_ids = torch.arange(len(data_object.source_encoded_input_ids[idx][0])).reshape(1,-1)
            bucketized_position_ids = torch.cat([torch.zeros((1, bucket_size - starting_input_len), dtype=torch.long), starting_position_ids], dim=1)
            
            data_list.append({'input_ids': bucketized_input_ids,
                            'attention_mask': bucketized_attention_mask,
                            'position_ids': bucketized_position_ids.squeeze(0)})
    
    return DataLoader(data_list, batch_size)