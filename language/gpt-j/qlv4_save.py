import yaml
from transformers import AutoConfig
import torch
import json
import quantization
import model_compressor
import joblib
import argparse

version='v3.12.1'
model_source="mlperf_submission"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model/", help="")
    parser.add_argument("--model_script_path", default="./quantization/model_script/Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-rope_lm-headint8.yaml", help="")
    parser.add_argument("--model_source", type = str, default = "mlperf_submission", help="the type of GPTJForCausalLM to use")
    parser.add_argument('--qformat_path', type = str, default=f'./quantization/output/{version}/{model_source}/qformat.yaml', help="")
    parser.add_argument('--qparam_path', type = str, default=f'./quantization/output/{version}/{model_source}/qparam.npy', help="")
    parser.add_argument('--qlv4_prefill_out_path', type = str, default=f'./quantization/output/{version}/{model_source}/prefill.bin', help="")
    parser.add_argument('--qlv4_decode_out_path', type = str, default=f'./quantization/output/{version}/{model_source}/decode.bin', help="")
    args = parser.parse_args()
    return args
    

#load model_script
def save_qlv4_model():
    args = get_args()
    torch_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device_type)

    
    
    ###hyperparameter###
    if args.model_source == "furiosa_llm_rope_rngd_gelu":
        from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
    elif args.model_source == "mlperf_submission":
        from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
    else:
        raise ValueError("other models are not considered.")
    config = AutoConfig.from_pretrained(args.model_path)
    model = GPTJForCausalLM.from_pretrained(args.model_path, config=config).to(device)
    
    model_generator = quantization.get_quant_model(model = model, 
                                                    calib_dataset_path = None, 
                                                    model_script_path = args.model_script_path, 
                                                    calib_without_padding = False,
                                                    recalibrate = False, 
                                                    qformat_path = args.qformat_path, 
                                                    qparam_path = args.qparam_path)

    if args.model_source == "furiosa_llm_rope_rngd_gelu":
        torch.save(model_generator.prefill_model.state_dict(), args.qlv4_prefill_out_path)
        torch.save(model_generator.decode_model.state_dict(), args.qlv4_decode_out_path)
    elif args.model_source == "mlperf_submission":
        torch.save(model_generator.prefill.state_dict(), args.qlv4_prefill_out_path)
        torch.save(model_generator.decode.state_dict(), args.qlv4_decode_out_path)

    print("success save qlv4 state dict")
    
    

if __name__ == "__main__":
    save_qlv4_model()
