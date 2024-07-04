import argparse
import os

import torch
import yaml
import model_compressor
from quantization.utils import random_seed, set_optimization
from quantization.get_quant_model import get_quant_model



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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument(
        "--quant_param_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--quant_format_path", help="quantization specifications for calibrated layers"
    )
    
    arser.add_argument(
        "--qlv4_prefill_output_path", help="quantization parameters for calibraed layers"
    )
    parser.add_argument(
        "--qlv4_decode_output_path", help="quantization specifications for calibrated layers"
    )

    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=-1,
        help="the number of layers"
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
    args = parser.parse_args()
    return args


def save_qlv4_model_statdict():
    args = get_args()
    random_seed()
    set_optimization(False)
    
    with open(args.quant_config_path, "r") as f:
        qconfig = yaml.safe_load(f)
    
    model = load_pytorch_model(args.model_source, args.model_path, args.n_layers)
    qlv4_model = get_quant_model(model, args,)
    if args.model_source == "mlperf_submission":
        torch.save(qlv4_model.prefill, args.qlv4_prefill_output_path)
        torch.save(qlv4_model.decode, args.qlv4_decode_output_path)
    else:
        torch.save(qlv4_model.prefill_model, args.qlv4_prefill_output_path)
        torch.save(qlv4_model.decode_model, args.qlv4_decode_output_path)
    

if __name__ == "__main__":
    save_qlv4_model_statdict()