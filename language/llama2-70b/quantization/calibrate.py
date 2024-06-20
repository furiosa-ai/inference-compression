import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import model_compressor
from .utils import get_kwargs, random_seed, set_optimization 

# Assume BLOCK_SIZE, NUM_BLOCKS, BUCKET_SIZE are fixed for now.
BLOCK_SIZE = 1
# bucket size would simply be a max value such as 2048 since we only provide one bucket
BUCKET_SIZE = 2048

def load_pytorch_model(model_source, model_path, use_gpu, n_layers):
    
    if use_gpu:
        assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
        
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    amp_dtype = torch.float32
    if model_source == 'furiosa_llm_original':
        from furiosa_llm_models.llama.symbolic.huggingface import LlamaForCausalLM 
    elif model_source == 'furiosa_llm_rope':
        from furiosa_llm_models.llama.symbolic.huggingface_rope import LlamaForCausalLM
    elif model_source == 'preallocated_concat_rope':
        from furiosa_llm_models.llama.symbolic.preallocated_concat_rope import LlamaForCausalLM
    elif model_source == 'mlperf_submission':
        from furiosa_llm_models.llama.symbolic.mlperf_submission import LlamaForCausalLM
    model_cls = LlamaForCausalLM
    
    if n_layers>0:
        from transformers import AutoConfig
        config_exp =  AutoConfig.from_pretrained(model_path)
        config_exp.num_hidden_layers = n_layers

        model = model_cls.from_pretrained(
            model_path, 
            config=config_exp
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


def cal_data_loader(model, model_source, data_path, batch_size, n_calib, max_seq_len=1024, use_generator_dataloader_with_unpacked_model=False):
    if not os.path.isfile(data_path):
        print("Calibration dataset {} not found. Please check that the path is correct".format(data_path))
    
    import pandas as pd
    calib_dataset = pd.read_pickle(data_path)
    
    input_tokens = calib_dataset['tok_input']
    max_length = 2048
    
    data_list = []
    if model_source == 'mlperf_submission' or use_generator_dataloader_with_unpacked_model:
        from furiosa_llm_models.generators.symbolic.llama_multi_gpu_paged_attention_optimized_generator import PagedAttentionGenerator
        generator = PagedAttentionGenerator(
                            model=model,
                            kv_dtype=torch.float32,
                            return_tensors=True,
                    )
    
    for input_token in input_tokens[:n_calib]:
        padding_size = padding_size = max_length - len(input_token)
        if model_source == 'mlperf_submission' or use_generator_dataloader_with_unpacked_model:
            data_list.append(generator.convert_data_for_prefill(input_ids=torch.tensor(input_token, dtype=torch.int32).view(1,-1), attention_mask=torch.ones((1,len(input_token)), dtype=torch.int32), use_generator_dataloader_with_unpacked_model=use_generator_dataloader_with_unpacked_model))
            generator.reset()
        else:
            data_list.append(
                {
                    "input_ids": pad(torch.tensor(input_token, dtype=torch.int32), (padding_size,0), value=2 ).view(1,-1).squeeze(0),
                    "attention_mask": pad(torch.ones((1,len(input_token)), dtype=torch.int32), (padding_size,0) ).squeeze(0),
                    'position_ids': pad(torch.arange(0, len(input_token), 1), (padding_size,0)),
                }
                
            )
            
    return DataLoader(data_list, batch_size=batch_size)


def calibrate(model, model_source, qconfig, qparam_path, qformat_path, calib_dataloader):
    if model_source == 'mlperf_submission':
        model = model.trace_decode()
    else:
        model, _,_ = model_compressor.helper.llama_custom_symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "position_ids"], 
            disable_check=True
            )


    model = model_compressor.create_quantsim_model(
        model,
        dataloader=calib_dataloader,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig),
    )

    
    model_compressor.calibrate(
        model,
        calib_dataloader=calib_dataloader,
        **get_kwargs(model_compressor.calibrate, qconfig),
    )

    model_compressor.save(
        model,
        qformat_out_path=qformat_path,
        qparam_out_path=qparam_path,
        **get_kwargs(model_compressor.save, qconfig),
    )

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend"
    )
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument(
        "--model_source", 
        type=str,
        choices=["furiosa_llm_rope",
                 "preallocated_concat_rope",
                 "mlperf_submission",
                 ], 
        help="choose model source"
    )
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    parser.add_argument("--calib_data_path", help="path to calibration data")
    parser.add_argument(
        "--torch_numeric_optim",
        action="store_true",
        help="use Pytorch numerical optimizaiton for CUDA/cudnn",
    )
    parser.add_argument(
        "--gpu", 
        type=bool, 
        default=True,
        help="use GPU instead of CPU for the inference"
    )
    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=-1,
        help="the number of layers"
    )
    parser.add_argument(
        "--n_calib", 
        type=int, 
        default=1000,
        help="the number of calibration samples"
    )
    
    parser.add_argument(
        "--use_generator_dataloader_with_unpacked_model", 
        action='store_true',
        help="use generator's dataloader to check qparam matching"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.backend == "pytorch":
        if not args.gpu:
            raise ValueError(
                "Inference on a device other than GPU is not suppurted yet."
            )
        model = load_pytorch_model(args.model_source, args.model_path, args.gpu, args.n_layers)

    else:
        raise ValueError("Unsupported backend: {:}".format(args.backend))

    random_seed()
    set_optimization(args.torch_numeric_optim)

    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    dataloader = cal_data_loader(
        model, args.model_source, args.calib_data_path, qconfig["calib_batch_size"], args.n_calib, use_generator_dataloader_with_unpacked_model=args.use_generator_dataloader_with_unpacked_model
    )
    calibrate(
        model,
        args.model_source,
        qconfig,
        args.quant_param_path,
        args.quant_format_path,
        dataloader,
    )


if __name__ == "__main__":
    main()