
import os
import random
from typing import Dict, List, Tuple
from dataset import Dataset
import torch
from torch.utils.data import DataLoader
from furiosa_llm_models.generators.packing import greedy_attention_packing
from .dataset_paged_attention import Dataset_for_paged_attention

def prepare_prefill_input_metadata(
        attention_mask: torch.Tensor, zero_block_index, available_block_indices, active_key_block_indices, active_value_block_indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        for prefill, valid_key_indices and valid_value_indices are none
        return (new_key_location, new_value_location)
        """
        new_key_location = []  # shape = (batch, bucket_size)
        new_value_location = []  # shape = (batch, bucket_size)
        for single_attention_mask in attention_mask:
            # for each attention_mask add zero block for padding
            block_indices = []
            for val in single_attention_mask:
                if val == 0:
                    # padding
                    block_indices.append(zero_block_index)
                else:
                    block_indices.append(available_block_indices.pop())

            active_key_block_indices.append(block_indices[:])
            active_value_block_indices.append(block_indices)

        new_key_location = torch.IntTensor(active_key_block_indices)
        new_value_location = torch.IntTensor(active_value_block_indices)

        return (
            new_key_location,
            new_value_location,
        )


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_key_location: torch.Tensor,
    new_value_location: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    return (packed_input_ids, causal_mask, packed_position_ids, logit_target_locations, packed_new_key_locatoin, packed_new_value_location)
    """  # noqa: E501
    (
        packed_attention_mask,
        packed_input_ids,
        causal_mask,
        logit_target_locations,
        packed_position_ids,
        packed_new_key_location,
        packed_new_value_location,
    ) = greedy_attention_packing(
        input_ids,
        attention_mask,
        new_key_location,
        new_value_location,
        pad_token_id=pad_token_id,
    )
    return (
        packed_input_ids,
        packed_attention_mask,
        causal_mask,
        packed_position_ids,
        logit_target_locations,
        packed_new_key_location,
        packed_new_value_location,
    )

def make_calib_dataloader_for_paged_attention_packed(
    calib_dataset_path, config, batch_size, bucket_size, total_block_space
):
    # input_ids, attention_mask, bucket_size, total_block_space):
    # The code is modified from furiosa-llm-models.generators.paged_attention_generator

    # There could be a bug associated with multi-batch calibration in mcp at the moment.
    assert batch_size == 1 

    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    block_indices, block_size, head, head_size = total_block_space[0][0].shape

    pad_token_id = config.pad_token_id
    for idx in range(len(data_object.source_encoded_input_ids)):
        # ----------- initial_settings -----------------
        active_key_block_indices = []
        active_value_block_indices = []
        available_block_indices = list(range(1, block_indices))
        zero_block_index = 0  # this is a special zero block

        starting_input_ids = data_object.source_encoded_input_ids[idx]
        starting_attention_mask = data_object.source_encoded_attn_masks[idx]
        batch_size, prompt_len = starting_input_ids.shape
        
        starting_position_ids = starting_attention_mask.long().cumsum(-1) - 1
        starting_position_ids.masked_fill_(starting_attention_mask == 0, 1)

        # ----------- adjust to bucket settings --------
        attention_mask = torch.zeros((batch_size, bucket_size), dtype=torch.int)
        attention_mask[:, :prompt_len] = starting_attention_mask

        input_ids = torch.zeros((batch_size, bucket_size), dtype=torch.int)
        input_ids[:, :prompt_len] = starting_input_ids

        position_ids = torch.zeros((batch_size, bucket_size), dtype=torch.long)
        position_ids[:, :prompt_len] = starting_position_ids

        (new_key_location, new_value_location) = prepare_prefill_input_metadata(attention_mask, zero_block_index, available_block_indices, active_key_block_indices, active_value_block_indices)

        (
                packed_input_ids,
                _packed_attention_mask,  # this attention mask if for debugging purpose
                causal_mask,
                packed_position_ids,
                logit_target_locations,
                new_key_location,
                new_value_location,
        ) = prepare_prefill_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                new_key_location=new_key_location,
                new_value_location=new_value_location,
                pad_token_id=pad_token_id,
        )  

        model_inputs = {
        "input_ids": packed_input_ids,
        "causal_mask": causal_mask,
        "position_ids": packed_position_ids,
        "new_key_location": new_key_location,
        "new_value_location": new_value_location,
        }

        data_list.append(model_inputs)
    
    dataset = Dataset_for_paged_attention(data_list, total_block_space)
    return DataLoader(dataset)



