import torch
from typing import List 
from torch.utils.data import DataLoader
from dataset import Dataset
from furiosa_llm_models.gptj.paged_attention_utils import InputMetadata

__all__ = ["make_calib_dataloader_for_paged_attention"]


def update_input_metadata(updated_attention_mask: List[List[int]], block_indices, block_size, bucket_size):
    new_key_locations = []
    new_value_locations = []
    batch_ids = []
    # since it's both padding
    # 일단 앞에서부터 하나씩 block 들을 만들고 block indices 를 부여하고 진짜인지 아닌지를 판단하기
    # 앞에서부터 하나씩 끊어서 block 을 만들면 됨

    available_block_indices = list(range(1, block_indices))
    active_key_block_indices = ([])  
    active_value_block_indices = ([]) 

    for batch_idx, single_attention_mask in enumerate(updated_attention_mask):
        batch_ids.append(torch.IntTensor([batch_idx]))
        new_key_location = []
        new_value_location = []
        split_blocks = [
            single_attention_mask[i : i + block_size]
            for i in range(0, len(single_attention_mask), block_size)
        ]

        active_key_block_indices.append([])
        active_value_block_indices.append([])

        # last_valid_key_block_idx = None
        # last_valid_value_block_idx = None
        # last_valid_token_idx = None

        for block in split_blocks:
            # x x 1 => then block is full
            # 1 x x => block is not full
            if sum(block) == 0:
                # then this is zero block

                new_key_location.append(torch.IntTensor([0]))
                new_value_location.append(torch.IntTensor([0]))

                active_key_block_indices[batch_idx].append(0)
                active_value_block_indices[batch_idx].append(0)
            else:
                # find the idx of last 1
                last_idx = 0
                for idx, val in enumerate(block):
                    if val == 1:
                        last_idx = idx

                new_key_block_idx = available_block_indices.pop()
                new_value_block_idx = available_block_indices.pop()

                new_key_location.append(torch.IntTensor([new_key_block_idx]))
                new_value_location.append(torch.IntTensor([new_value_block_idx]))

                active_key_block_indices[batch_idx].append(new_key_block_idx)
                active_value_block_indices[batch_idx].append(new_value_block_idx)

                # last_valid_key_block_idx = new_key_block_idx
                # last_valid_value_block_idx = new_value_block_idx
                # last_valid_token_idx = last_idx

        # self.valid_block_meta.append(
        #     (
        #         (last_valid_key_block_idx, last_valid_token_idx),
        #         (last_valid_value_block_idx, last_valid_token_idx),
        #     )
        # )

        new_key_locations.append(torch.unsqueeze(torch.cat(new_key_location), 0))
        new_value_locations.append(torch.unsqueeze(torch.cat(new_value_location), 0))

    new_key_locations = torch.cat(new_key_locations)
    new_value_locations = torch.cat(new_value_locations)
    batch_ids = torch.cat(batch_ids)

    input_metadata = InputMetadata(
        key_cache_idx=active_key_block_indices,
        value_cache_idx=active_value_block_indices,
        new_key_location=new_key_locations,
        new_value_location=new_value_locations,
        block_max_seq_len=int(bucket_size / block_size),
        block_size=block_size, 
        is_prefill=True,
    )

    input_metadata = [input_metadata.new_key_location.squeeze(0), input_metadata.new_value_location.squeeze(0), input_metadata.bucket_size, input_metadata.valid_key_indices.squeeze(0), input_metadata.valid_value_indices.squeeze(0)] 

    return input_metadata

def make_calib_dataloader_for_paged_attention(calib_dataset_path, batch_size, bucket_size, total_block_space):
    # input_ids, attention_mask, bucket_size, total_block_space):
    # The code is modified from furiosa-llm-models.generators.paged_attention_generator

    #There could be a bug associated with multi-batch calibration in mcp at the moment. 
    assert batch_size == 1 
    # batch_size = 2
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    block_indices, block_size, head, head_size = total_block_space[0][0].shape
    for idx in range(len(data_object.source_encoded_input_ids)):
        starting_input_ids = data_object.source_encoded_input_ids[idx]
        final_input_ids = starting_input_ids
        starting_attention_mask =  data_object.source_encoded_attn_masks[idx]
        batch_size, starting_input_len = starting_input_ids.shape
        bucketized_attention_mask = torch.zeros((batch_size, bucket_size), dtype=torch.int)
        bucketized_attention_mask[:, :starting_input_len] = starting_attention_mask

        #make position_ids
        starting_position_id = []
        for single_attention_mask in starting_attention_mask.tolist():
            # find the first 1, then every before that is 1
            # and new indexing from 0 begins there
            # position ids is very similar to attention mask
            # if bucket size = 8
            # [[x x a b]
            #  [a b c d]
            #  [x x x a]]
            # then, position id would be d
            # [[1 1 0 1 2 3]
            #  [0 1 2 3 4 5]
            #  [1 1 1 0 1 2]]
            target_idx = 0
            for idx, value in enumerate(single_attention_mask):
                if value == 1:
                    target_idx = idx
                    break

            single_attention_mask[:target_idx] = [1] * target_idx
            single_attention_mask[target_idx:] = list(
                range(len(single_attention_mask) - target_idx)
            )
            single_position_id = torch.cat(
                [
                    torch.LongTensor(single_attention_mask).reshape(1, -1),
                    torch.zeros((1, bucket_size - starting_input_len), dtype=torch.long),
                ],
                dim=1,
            )
            starting_position_id.append(single_position_id)

        starting_position_ids = torch.cat(starting_position_id, dim=0)

        bucketized_input_ids = torch.zeros((batch_size, bucket_size), dtype=torch.int)
        bucketized_input_ids[:, :starting_input_len] = starting_input_ids

        input_metadata = update_input_metadata(bucketized_attention_mask.tolist(), block_indices, block_size, bucket_size)

        model_inputs= {"input_metadata": input_metadata,
                        "input_ids": bucketized_input_ids.squeeze(0), 
                        "attention_mask": bucketized_attention_mask.squeeze(0), 
                        "position_ids": starting_position_ids.squeeze(0), 
                        "past_key_values": total_block_space, 
                    }
        data_list.append(model_inputs)
    

    return DataLoader(data_list, batch_size)
    




        





