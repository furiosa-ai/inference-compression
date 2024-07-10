import argparse
import os
import gc

import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import model_compressor
from quantization.utils import get_kwargs, random_seed, set_optimization
from quantization.get_quant_model import get_quant_model
from quantization.calibrate import make_calib_dataloader, calibrate
import pickle
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers import AutoTokenizer

def check_logits(
    golden_model_file_path,
    comparison_model_file_path,
    mcm_module_name,
    is_decode,
):

    golden_tensor_list = load_all_tensors_from_pickle(golden_model_file_path, mcm_module_name)
    comparison_tensor_list = load_all_tensors_from_pickle(comparison_model_file_path, mcm_module_name)
    

    assert len(golden_tensor_list) == len(comparison_tensor_list)
    
    
    for idx in range(len(golden_tensor_list)):
        valid_seq_len = golden_tensor_list[idx].shape[1] if not is_decode else 1
        
        if golden_tensor_list[idx].shape[0] != comparison_tensor_list[idx].shape[0]: 
            #If true, packing would have been applied in furiosa-llm-generator due to the short length of input_ids
            is_successful = torch.equal(golden_tensor_list[idx][:, -valid_seq_len:, :][0].unsqueeze(0), comparison_tensor_list[idx][:, -valid_seq_len:, :])
        else:
            is_successful = torch.equal(golden_tensor_list[idx][:, -valid_seq_len:, :], comparison_tensor_list[idx][:, -valid_seq_len:, :])

        if not is_successful:
            raise ValueError("Logits comparison test failed.")
        
    return True

# Assume BLOCK_SIZE, NUM_BLOCKS, BUCKET_SIZE are fixed for now.
BLOCK_SIZE = 1
# bucket size would simply be a max value such as 2048 since we only provide one bucket
EARLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 2
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 1
NUM_BEAMS = 1
DO_SAMPLE = False
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.float32
QUANT_KV_DTYPE = torch.int8
BUCKET_SIZE = 2048

gen_kwargs = {
    "early_stopping": True,
    "min_new_tokens": 1,
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

def load_pytorch_model(model_source, model_path, n_layers):
    amp_dtype = torch.float32
    if model_source == 'furiosa_llm_rope':
        from furiosa_llm_models.llama.symbolic.huggingface_rope import LlamaForCausalLM
    elif model_source == 'mlperf_submission':
        from furiosa_llm_models.llama.symbolic.mlperf_submission import LlamaForCausalLM
    model_cls = LlamaForCausalLM
    
    if n_layers>0:
        from transformers import AutoConfig
        config_exp =  AutoConfig.from_pretrained(model_path)
        config_exp.num_hidden_layers = n_layers

        model = model_cls.from_pretrained(
            model_path, 
            config=config_exp,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=amp_dtype
        )
    else:
        model = model_cls.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=amp_dtype
            )
    print("Loaded model")

    model.eval()
    model = model.to(memory_format=torch.channels_last)
    
    return model


def gen_test_data(data_path, n_data=1):
    if not os.path.isfile(data_path):
        print("Dataset {} not found. Please check that the path is correct".format(data_path))
    
    import pandas as pd
    eval_dataset = pd.read_pickle(data_path)
    
    input_tokens = eval_dataset['tok_input']
    
    data_list = []
    
    for input_token in input_tokens[:n_data]:
        test_data = {
                "input_ids": torch.tensor(input_token, dtype=torch.int32, device='cuda').view(1,-1),
                "attention_mask": torch.ones((1,len(input_token)), dtype=torch.int32, device='cuda'),
            }
        data_list.append(test_data)
            
    return data_list


def load_all_tensors_from_pickle(file_path, mcm_module_name):
    tensor_list = []
    with open(file_path, "rb") as file:
        while True:
            try:
                result_tensor = pickle.load(file)
                layer_name = next(iter(result_tensor))
                if mcm_module_name in layer_name:
                    tensor_list.append(result_tensor[mcm_module_name]["output_before_rounding"])
                    
            except EOFError:
                break
    return tensor_list
            




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--dataset_path", help="path to eval data")
    parser.add_argument("--calib_data_path", help="path to calib data")

    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=-1,
        help="the number of layers"
    )
    
    parser.add_argument(
        "--n_calib", 
        type=int, 
        default=20,
        help="the number of calib data"
    )

    parser.add_argument(
        "--model_source", 
        type=str,
        choices=["furiosa_llm_rope",
                 "preallocated_concat_rope",
                 "mlperf_submission",
                 ], 
        help="choose model source"
    )
    parser.add_argument("--weighted_op_emul_dtype", type=str, default="fp64", help="set emulation type of weighted operators")
    args = parser.parse_args()
    return args


def test_model_equivalence():
    args = get_args()
    random_seed()
    set_optimization(False)
    
    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    
    # create golden model 
    args.model_source = "furiosa_llm_rope"
    model = load_pytorch_model(args.model_source, args.model_path, args.n_layers)
    
    # gen calib data and do calibrate
    if args.n_calib > 0:
        if 'qlevel' in qconfig:
            qconfig['qlevel'] = 2
            
        dataloader = make_calib_dataloader(model, args.model_source, args.calib_data_path, qconfig["calib_batch_size"], args.n_calib)
        calibrate(
            model,
            args.model_source,
            qconfig,
            args.quant_param_path,
            args.quant_format_path,
            dataloader,
        )

    

    
    # gen test_data
    test_data_list = gen_test_data(args.dataset_path, n_data = 2)


    # create quant golden model and activate dump mode
    golden_file_path = "./golden_dump"
    golden_model = get_quant_model(model, args,)
    model_compressor.set_model_to_dump_golden_model(
        golden_file_path + '_prefill',
        golden_model.prefill_model,
        dumping_range='lm_head',
        dumping_mode='only-in-out', 
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True,)

    
    model_compressor.set_model_to_dump_golden_model(
        golden_file_path + '_decode',
        golden_model.decode_model,
        dumping_range="lm_head",    
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True)
        # generate
    with torch.no_grad():
        for test_data in test_data_list:
            output = golden_model.generate(**test_data, **gen_kwargs)
    

    del golden_model
    del model
    gc.collect()
    
    # create mlperf model
    args.model_source = "mlperf_submission"
    model = load_pytorch_model(args.model_source, args.model_path, args.n_layers)
    max_position_embeddings = getattr(model, "max_position_embeddings", None)
    # create quant golden model and activate dump mode
    mlperf_path = "./mlperf_dump"
    mlperf_model = get_quant_model(model, args, immigrate_qparams=True)
    model_compressor.set_model_to_dump_golden_model(
        mlperf_path + '_prefill',
        mlperf_model.prefill,
        dumping_range="lm_head",    
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True)
    
    model_compressor.set_model_to_dump_golden_model(
        mlperf_path + '_decode',
        mlperf_model.decode,
        dumping_range="lm_head",    
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=1024,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # generate

    max_seq_len = 1024

    with torch.no_grad():
        for test_data in test_data_list:
            input_ids_tensor = []
            input_masks_tensor = []


            input_ids_tensor.append(pad(test_data['input_ids'],
                                        (max_seq_len - test_data['input_ids'].shape[-1], 0, 0, 0),
                                        value=tokenizer.pad_token_id))

            input_masks_tensor.append(pad(test_data['attention_mask'],
                                            (max_seq_len - test_data['attention_mask'].shape[-1], 0, 0, 0),
                                            value=0))


            input_ids_tensor = torch.cat(input_ids_tensor)
            input_masks_tensor = torch.cat(input_masks_tensor)


        
            logits_processor = LOGITS_PROCESSOR(
                    input_ids_tensor.shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
                )

            stopping_criteria = STOPPING_CRITERIA(
                    MAX_LENGTH,
                    max_position_embeddings,
                )

            output = mlperf_model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    max_length=MAX_LENGTH,
                    pad_token_id=PAD_TOKEN_ID,
                    eos_token_id=EOS_TOKEN_ID,
                    return_dict_in_generate=RETURN_DICT_IN_GENERATE,
                    kv_dtype=QUANT_KV_DTYPE,
                    bucket_size=BUCKET_SIZE,
                )
            # output = mlperf_model.generate(**test_data, max_length=seq_len+gen_kwargs["max_new_tokens"])

    del mlperf_model
    del model
    gc.collect()

    check_logits(golden_file_path+'_prefill', mlperf_path+'_prefill', mcm_module_name = 'lm_head', is_decode = False)
    check_logits(golden_file_path+'_decode', mlperf_path+'_decode', mcm_module_name = 'lm_head', is_decode = True)
    print("Logits comparison test passed.")
    
if __name__ == "__main__":
    test_model_equivalence()