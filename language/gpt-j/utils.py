import json
import os
import io
import torch
import numpy as np
import random 
from torch.utils.data import DataLoader, Dataset
import collections
import copy
import model_compressor
import furiosa_llm_models

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_optimization(args):
    if args.torch_optim == 'default':
        return
    elif args.torch_optim == 'none':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, 'opt_einsum'):
            torch.backends.opt_einsum.enabled = False
    else:
        raise ValueError(f"Wrong argument value for '--torch_optim': {args.torch_optim}")

    return

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

###########
def transform_past_key_shape(tensor, shape):
    if isinstance(tensor, torch.Tensor):
        new_tensor = torch.zeros(shape).to(tensor.device)
        del tensor
        return new_tensor
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(transform_past_key_shape(d, shape) for d in tensor)
    else:
        raise TypeError("Unsupported tensor type")

def postproc_for_PagedAttentionGeneratorBeamSearch(generator, cnt, **input_kwargs):
    batch_size=1
    block_size=1
    num_beams=1

    input_kwargs['input_ids'] = input_kwargs['input_ids'][0,:]
    input_kwargs['causal_mask'] = input_kwargs['causal_mask'][0,:]
    input_kwargs['position_ids'] = input_kwargs['position_ids'][0,:]
    input_kwargs['new_key_location'] = input_kwargs['new_key_location'][0,:]
    input_kwargs['new_value_location'] = input_kwargs['new_value_location'][0,:]

    past_key_values = input_kwargs.get('past_key_values')
    # proprocessing
    max_prompt_len = generator.bucket_size - generator.max_new_tokens
    num_blocks = batch_size * num_beams * 2 * (max_prompt_len) * block_size + 1 
    shape_info = list(past_key_values[0][0].shape)
    shape_info[0] = num_blocks
    if cnt == 0:
        input_kwargs['past_key_values'] = transform_past_key_shape(past_key_values, shape_info)
    else:
        del input_kwargs['past_key_values']
    torch.cuda.empty_cache()
    return input_kwargs
