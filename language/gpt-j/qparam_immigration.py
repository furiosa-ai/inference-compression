import yaml
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
import json
import quantization
from quantization import calibration_utils
import model_compressor
from dataset import Dataset
import joblib




gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 4, 
}

    
    

#load model_script
def compare_model_outputs():
    torch_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device_type)

    from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
    model_path = './model'
    model_config = AutoConfig.from_pretrained('./ci_test_file/config.json')
    # model_config.n_layer = 4
    #To test without downloading the MLPerf model, load the model as below.
    #config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
    #golden_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", config=config).to(device)
    
    calib_dataset_path = './ci_test_file/calibration_dataset_20.json'
    model_script_path = './ci_test_file/model_script.yaml'
    qformat_path = './ci_test_file/qformat_v3.12_gptj_golden.yaml'
    qparam_path = './ci_test_file/qparam_v3.12_gptj_golden.npy'
        
    
           
    from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
    submission_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)
    
    
    submission_generator = quantization.get_quant_model_immigration(model = submission_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = False, 
                                                     qformat_path = qformat_path, 
                                                     qparam_path = qparam_path,
                                                     immigrate_qparams = True)
    

    exit()

    print("gptj forward ci test is passed")
    
    

if __name__ == "__main__":
    compare_model_outputs()
