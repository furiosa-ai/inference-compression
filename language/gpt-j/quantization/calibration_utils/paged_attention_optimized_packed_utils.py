
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from dataset import Dataset

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


def greedy_attention_packing(
    input_ids: torch.Tensor,
    bucketized_attention_mask: torch.Tensor,
    new_key_location: torch.Tensor,
    new_value_location: torch.Tensor,
    eos_token_id: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[List[int]],
    torch.Tensor,
    torch.Tensor,
]:
    """
    return (packed_attention_mask, packed_input_ids, causal_mask, packed_position_ids, logit_target_locations, packed_new_key_location, packed_new_value_location)
    """  # noqa: E501
    assert input_ids.shape == bucketized_attention_mask.shape
    assert bucketized_attention_mask.shape == new_key_location.shape
    assert bucketized_attention_mask.shape == new_value_location.shape

    logit_target_locations = []
    (original_batch, bucket_size) = bucketized_attention_mask.shape

    # split attention mask by batch
    # print("bucketized attention mask: ", bucketized_attention_mask)
    batch_real_len = []
    for single_batch in bucketized_attention_mask:
        num_real_token = single_batch.sum().item()
        batch_real_len.append(num_real_token)

    # find real tokens
    # first convert all padding tensors to 0
    converted_input_ids = torch.where(input_ids == eos_token_id, 0, input_ids)
    non_zero_indices = converted_input_ids.nonzero().tolist()

    # print("non zero indices: ", non_zero_indices)
    # print("batch real len: ", batch_real_len)

    real_locations = []
    for i, real_len in enumerate(batch_real_len):
        locations = [non_zero_indices.pop(0)[1] for _ in range(real_len)]
        start = locations[0]
        end = locations[-1] + 1
        real_locations.append((i, start, end))

    marker = bucket_size
    target_locations: List[List[Tuple[int, int]]] = []  # List of List
    temp_indices = []
    for i in range(original_batch):
        cur_len = batch_real_len[i]
        if marker - cur_len < 0:
            # we cannot pack so start a new row
            target_locations.append(temp_indices)
            temp_indices = []
            marker = bucket_size

        temp_indices.append((marker - cur_len, marker))
        marker -= cur_len

    # push the last row into the target locations
    target_locations.append(temp_indices)

    packed_batch_size = len(target_locations)

    # initialize attention mask
    packed_shape = (packed_batch_size, bucket_size)

    packed_attention_mask = torch.zeros(packed_shape, dtype=torch.bool)
    packed_input_ids = torch.zeros(packed_shape, dtype=torch.int32)
    packed_new_key_location = torch.zeros(packed_shape, dtype=torch.int32)
    packed_new_value_location = torch.zeros(packed_shape, dtype=torch.int32)
    position_ids = torch.ones(packed_shape, dtype=torch.long)

    # initialize causal mask
    causal_mask = torch.zeros((packed_batch_size, bucket_size, bucket_size), dtype=torch.bool)

    # fill the new attention mask and mark the logit locations
    for index, target_location in enumerate(target_locations):
        # record new target locations
        logit_target_location = []
        for start, end in target_location:
            (original_index, original_start, original_end) = real_locations.pop(0)
            packed_attention_mask[index][start:end] = True
            packed_input_ids[index][start:end] = input_ids[original_index][
                original_start:original_end
            ]
            packed_new_key_location[index][start:end] = new_key_location[original_index][
                original_start:original_end
            ]
            packed_new_value_location[index][start:end] = new_value_location[original_index][
                original_start:original_end
            ]
            position_ids[index][start:end] = torch.arange(end - start)
            logit_target_location.append(end - 1)
            # print(
            #     "index: {index}, start: {start}, end: {end}".format(
            #         index=index, start=start, end=end
            #     )
            # )
            causal_mask[index][start:end, start:end] = torch.tril(
                torch.ones((end - start, end - start), dtype=torch.bool)
            )
        logit_target_locations.append(logit_target_location)

    return (
        packed_attention_mask,
        packed_input_ids,
        causal_mask,
        logit_target_locations,
        position_ids,
        packed_new_key_location,
        packed_new_value_location,
    )

def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_key_location: torch.Tensor,
    new_value_location: torch.Tensor,
    eos_token_id: int,
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
        eos_token_id=eos_token_id,
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

def prepare_decode_input_metadata(active_key_block_indices, active_value_block_indices, available_block_indices
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return (new_key_location, new_value_location, valid_key_indices, valid_valie_indices)
        """
        new_key_location = []  # shape = (batch, 1)
        new_value_location = []  # shape = (batch, 1)
        valid_key_indices = []  # shape = (batch*(bucket_size -1))
        valid_value_indices = []  #
        for key_batch, value_batch in zip(
            active_key_block_indices, active_value_block_indices
        ):
            valid_key_indices.extend(key_batch[:-1])
            valid_value_indices.extend(value_batch[:-1])

            # we use same block idx for key and value here
            new_block_idx = available_block_indices.pop()

            key_batch[-1] = new_block_idx
            value_batch[-1] = new_block_idx

            new_key_location.append([new_block_idx])
            new_value_location.append([new_block_idx])

        new_key_location = torch.IntTensor(new_key_location)
        new_value_location = torch.IntTensor(new_value_location)
        valid_key_indices = torch.IntTensor(valid_key_indices)
        valid_value_indices = torch.IntTensor(valid_value_indices)

        return (
            new_key_location,
            new_value_location,
            valid_key_indices,
            valid_value_indices,
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
    eos_token_id = config.eos_token_id
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
                eos_token_id=eos_token_id,
        )  

        model_inputs = {
        "input_ids": packed_input_ids,
        "causal_mask": causal_mask,
        "position_ids": packed_position_ids,
        "past_key_values": total_block_space,
        "new_key_location": new_key_location,
        "new_value_location": new_value_location,
        }

        data_list.append(model_inputs)

    return DataLoader(data_list)
