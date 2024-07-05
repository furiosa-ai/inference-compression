import yaml
from transformers import AutoConfig
import torch
import json
import quantization
import model_compressor
import joblib
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument("--model_config", default="./ci_test_file/config.json", help="")
    parser.add_argument("--model_script_path", default="./quantization/model_script/Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-rope_lm-headint8.yaml", help="")
    parser.add_argument("--model_source", type = str, default = "mlperf_submission", help="the type of GPTJForCausalLM to use")
    parser.add_argument('--qformat_path', type = str, default="./quantization/output/qformat_Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-mlperf_submission.yaml", help="")
    parser.add_argument('--qparam_path', type = str, default="./quantization/output/qparam_Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-mlperf_submission.npy", help="")
    parser.add_argument('--qlv4_prefill_out_path', type = str, default='./quantization/model_script/prefill.bin', help="")
    parser.add_argument('--qlv4_decode_out_path', type = str, default='./quantization/model_script/decode.bin', help="")
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
    config = AutoConfig.from_pretrained(args.model_config)
    model = GPTJForCausalLM.from_pretrained(args.model_path, config=config).to(device)
    
    model_genearator = quantization.get_quant_model(model = model, 
                                                    model_script_path = args.model_script_path, 
                                                    recalibrate = False, 
                                                    qformat_path = args.qformat_path, 
                                                    qparam_path = args.qparam_path)

    if args.model_source == "furiosa_llm_rope_rngd_gelu":
        torch.save(model_geneartor.prefill_model.state_dict(), args.qlv4_prefill_out_path)
        torch.save(model_genearator.decode_model.state_dict(), args.qlv4_decode_out_path)
    elif args.model_source == "mlperf_submission":
        torch.save(model_geneartor.prefill.state_dict(), args.qlv4_prefill_out_path)
        torch.save(model_genearator.decode.state_dict(), args.qlv4_decode_out_path)

    print("success save qlv4 state dict")
    
    

if __name__ == "__main__":
    save_qlv4_model()
