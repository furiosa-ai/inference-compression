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



#########################
def recursive_move_to_device(obj, device, seen=None):
    if seen is None:
        seen = set()

    if id(obj) in seen:
        return obj
    seen.add(id(obj))

    if isinstance(obj, collections.abc.Mapping):
        for key, value in obj.items():
            obj[key] = recursive_move_to_device(value, device, seen)
    elif isinstance(obj, tuple):
        obj = list(obj)
        for i, item in enumerate(obj):
            obj[i] = recursive_move_to_device(item, device, seen)
        obj = tuple(obj)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = recursive_move_to_device(item, device, seen)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)

    return obj


class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphModuleWrapper(torch.fx.GraphModule):
    def __init__(self, model, device):
        super().__init__(model, model.graph)
        self.model = model

        if hasattr(model, 'config'):
            self.config = model.config
        if hasattr(model, 'concrete_args'):
            self.concrete_args = model.concrete_args

        self.collected_inputs = {}
        self.device = device

    def __call__(self, *args, **forward_kwargs):
        if len(args) != 0:
            raise ValueError("Unexpected Error")

        self.collected_inputs = {}  # 매 Input 마다 빈 Dict로 Initialize
        # Collect the inputs
        for key, value in forward_kwargs.items():
            self.collected_inputs[key] = value

        self.collected_inputs = recursive_move_to_device(self.collected_inputs, self.device)
        #  model(**forward_kwargs)는 수행하지 않도록 의도적 에러 호출
        raise CustomError("Generating sample calibration data...")


class SampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def convert_json_to_list_set(
        dataset_path,
        batch_size=1,
        pad_val=1,
        pad_max=196,
        num_splits=1,split_idx=0
    ):

    list_data_dict = jload(dataset_path)
    if num_splits > 1:
        n_splited_data = int(len(list_data_dict)/num_splits)
        start_idx = split_idx*n_splited_data
        end_idx= (split_idx+1)*n_splited_data if split_idx!=num_splits-1 else len(list_data_dict) + 1
        list_data_dict = list_data_dict[start_idx:end_idx]

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [prompt_input.format_map(example) for example in list_data_dict]
    return sources

def get_model_compressor_generator(model, prefill_model, decode_model):
    model_type = type(model)
    model = {
        "prefill_model": prefill_model,
        "decode_model": decode_model,
    }
    input_names = {
        "prefill_input_names": prefill_model.input_names,
        "decode_input_names": decode_model.input_names,
    }
    concrete_args = {
        "prefill_concrete_args": prefill_model.concrete_args,
        "decode_concrete_args": decode_model.concrete_args,
    }

    generator_class = model_compressor.helper.QuantCausalLM(
        model, model_type, input_names, concrete_args
    )
    return generator_class

def paged_attention_rope_tracer(model):
    model_for_calib, _, _ = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = False, disable_check=True)    
    return model_for_calib, model_for_calib

def create_sample_calib_dataloader_from_generator(
    sample_dataset_path,
    vanila_model,
    model_name,
    generator_class,
    tokenizer_class,
    device='cpu',
    padding_side="left",
    max_length=100,
    min_new_tokens=10,
    bucket_size=256,
    model_max_length=2048,
    use_fast=False,
):
    '''
    mlperf_submission 모델: GPT-J, LLAMA 테스트 완료
    (ex
        from furiosa_llm_models.generators.paged_attention_optimized_generator_beam_search_optimized import PagedAttentionGeneratorBeamSearch,
        from furiosa_llm_models.generators.paged_attention_optimized_generator import PagedAttentionGenerator
        from furiosa_llm_models.generators.symbolic.llama_multi_gpu_paged_attention_optimized_generator import PagedAttentionGenerator
    )
    '''
    input_prompts = convert_json_to_list_set(sample_dataset_path)
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        use_fast=use_fast,
    )
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token

    # 1. trace model
    vanila_traced_model = vanila_model.trace_all()
    prefill_model = vanila_traced_model['prefill']
    decode_model = vanila_traced_model['decode']
    
    # 2. wrap vanila model
    wrapped_prefill_model = GraphModuleWrapper(prefill_model, device)
        
    # 3. create sample data from tokenizer
    prefill_data_list = []
    for idx, input_prompt in enumerate(input_prompts):
        try:
            if generator_class == None: 
                # EX) model_compressor.helper.QuantCausalLM
                wrapped_prefill_model.input_names = prefill_model.input_names
                vanila_generator = get_model_compressor_generator(vanila_model, wrapped_prefill_model, decode_model)
            else:
                vanila_generator = generator_class(
                    prefill=wrapped_prefill_model, decode=decode_model
                )
            encoded_input = tokenizer(input_prompt, return_tensors="pt").to(device)
            vanila_generator.generate(
                **encoded_input, max_length=max_length, min_new_tokens=min_new_tokens
            )

        except CustomError as e:
            print(f'{e} \t {idx+1}th data generated.')

        prefill_data_list.append(copy.deepcopy(wrapped_prefill_model.collected_inputs))

    # 4. sample_calib_dataloader 생성
    dataset = SampleDataset(prefill_data_list)
    return DataLoader(dataset)

