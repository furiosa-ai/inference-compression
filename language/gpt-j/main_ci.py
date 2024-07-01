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
import pickle




gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": 4, 
}



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
    



#load model_script
def compare_model_outputs():
    torch_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device_type)

    from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
    model_path = './model'
    model_config = AutoConfig.from_pretrained('./ci_test_file/config.json')
    golden_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)
    
    #To test without downloading the MLPerf model, load the model as below.
    # model_path = "EleutherAI/gpt-j-6B"
    # model_config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
    # golden_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)


    calib_dataset_path = './ci_test_file/calibration_dataset.json'
    evaluation_dataset_path = './ci_test_file/evaluation_dataset.json'
    model_script_path = './ci_test_file/model_script.yaml'
    qformat_path = './ci_test_file/golden_qformat.yaml'
    qparam_path = './ci_test_file/golden_qparam.npy'
    
    
    golden_model_generator = quantization.get_quant_model(model = golden_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = True, 
                                                     qformat_path = qformat_path, 
                                                     qparam_path = qparam_path)
    
    
           
    from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
    submission_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)

    submission_generator = quantization.get_quant_model(model = submission_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = False, 
                                                     qformat_path = qformat_path, 
                                                     qparam_path = qparam_path,
                                                     immigrate_qparams = True,
                                                     gen_kwargs = gen_kwargs)
    
    
    #Turn on mcp dump
    model_compressor.set_model_to_dump_golden_model(
            './ci_test_file/golden_prefill_logits.pkl',
            golden_model_generator.prefill_model,
            dumping_range='lm_head',
            dumping_mode='only-in-out', 
            qlv4_skip_output_rounding=False,
            dumping_before_rounding=True,
            dump_in_append_mode=True,)
    
    model_compressor.set_model_to_dump_golden_model(
            './ci_test_file/golden_decode_logits.pkl',
            golden_model_generator.decode_model,
            dumping_range="lm_head",    
            dumping_mode="only-in-out",
            qlv4_skip_output_rounding=False,
            dumping_before_rounding=True,
            dump_in_append_mode=True)
    
    model_compressor.set_model_to_dump_golden_model(
            './ci_test_file/submission_prefill_logits.pkl',
            submission_generator.prefill,
            dumping_range='lm_head',
            dumping_mode='only-in-out', 
            qlv4_skip_output_rounding=False,
            dumping_before_rounding=True,
            dump_in_append_mode=True)
    
    model_compressor.set_model_to_dump_golden_model(
        './ci_test_file/submission_decode_logits.pkl',
        submission_generator.decode,
        dumping_range="lm_head",     #layer_name to dump
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True        #enable append mode
        )
    
    


    validation_dataset = Dataset(dataset_path = evaluation_dataset_path)
    
    #Due to the issue associated with the generator, the tested token and gen_kwargs were adjusted to avoid RunTimeError. 
    # After the issue is resolved, the test will be performed with mlperf validation data and gen_kwargs.
    for idx in range(len(validation_dataset.sources)):
        
        input_batch = dict()
        input_batch['input_ids'] = validation_dataset.source_encoded_input_ids[idx].to(device)
        input_batch['attention_mask'] = validation_dataset.source_encoded_attn_masks[idx].to(device)
        seq_len = input_batch['input_ids'].shape[1]
        output_batch_golden = golden_model_generator.generate(**input_batch, **gen_kwargs, pad_token_id = model_config.eos_token_id)
        output_batch_submission = submission_generator.generate(**input_batch, **gen_kwargs)

    
   
    check_logits(golden_model_file_path='./ci_test_file/golden_prefill_logits.pkl',
                    comparison_model_file_path ='./ci_test_file/submission_prefill_logits.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = False)

    check_logits(golden_model_file_path='./ci_test_file/golden_decode_logits.pkl',
                    comparison_model_file_path ='./ci_test_file/submission_decode_logits.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = True)


    print("gptj forward ci test is passed")
    
    

if __name__ == "__main__":
    compare_model_outputs()
