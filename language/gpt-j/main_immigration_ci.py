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

def is_logit_same(
    golden_model_file_path,
    comparison_model_file_path,
    mcm_name_to_check,
    decode=False,
):
    import pickle

    import torch

    comparison_file = open(comparison_model_file_path, "rb")

    read_golden_file = True

    with open(golden_model_file_path, "rb") as golden_file:
        while read_golden_file:
            try:
                golden_result = pickle.load(golden_file)
                golden_layer_name = next(iter(golden_result))

                if (
                    mcm_name_to_check is not None
                    and not mcm_name_to_check in golden_layer_name
                ):
                    continue

            except EOFError:
                break


        while True:
            comparison_result = pickle.load(comparison_file)
            comparison_layer_name = next(iter(comparison_result))

            if golden_layer_name in comparison_layer_name:
                read_golden_file = False
                break
            
    golden_result = golden_result[golden_layer_name]
    comparison_result = comparison_result[comparison_layer_name]


    valid_golden_output = golden_result["output_before_rounding"]
    valid_seq_len = valid_golden_output.shape[1] if not decode else 1
    valid_comparison_output = comparison_result["output_before_rounding"][:, -valid_seq_len:, :]
    
    if not torch.equal(valid_golden_output[0].unsqueeze(0), valid_comparison_output[0].unsqueeze(0)):
        raise ValueError("Logits comparison test failed.")
    
    return True
    
    
    

#load model_script
def compare_model_outputs():
    torch_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device_type)

    from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
    model_path = './model'
    model_config = AutoConfig.from_pretrained('./ci_test_file/config.json')
    model_config.n_layer = 4
    golden_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)
    
    #To test without downloading the MLPerf model, load the model as below.
    #config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
    #golden_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", config=config).to(device)
    
    
    calib_dataset_path = './ci_test_file/calibration_dataset_20.json'
    evaluation_dataset_path = './ci_test_file/evaluation_dataset_short.json'
    model_script_path = './ci_test_file/model_script.yaml'
    qformat_path = './ci_test_file/qformat_Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-rope.yaml'
    qparam_path = './ci_test_file/qparam_Qlevel4_RGDA0-W8A8KV8-PTQ-SMQ-rope.npy'
    
    
    golden_model_generator = quantization.get_quant_model(model = golden_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = False, 
                                                     qformat_path = qformat_path, 
                                                     qparam_path = qparam_path)
    
    
           
    from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
    submission_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)
    
    New_qformat_path = './ci_test_file/new_qformat.yaml'
    New_qparam_path = './ci_test_file/new_qparam.npy'
    
    submission_generator = quantization.get_quant_model(model = submission_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = False, 
                                                     qformat_path = New_qformat_path, 
                                                     qparam_path = New_qparam_path,
                                                     immigrate_qparams = False)
    
    
    #Turn on dumping 
    model_compressor.set_model_to_dump_golden_model(
                './ci_test_file/golden_prefill_logits.pkl',
                golden_model_generator.prefill_model,
                dumping_range='qlv4_linear',
                dumping_mode='only-in-out', 
                qlv4_skip_output_rounding=False,
                dumping_before_rounding=True,
            )
    model_compressor.set_model_to_dump_golden_model(
                './ci_test_file/golden_decode_logits.pkl',
                golden_model_generator.decode_model,
                dumping_range='qlv4_linear',
                dumping_mode='only-in-out', 
                qlv4_skip_output_rounding=False,
                dumping_before_rounding=True,
            )
    model_compressor.set_model_to_dump_golden_model(
                './ci_test_file/submission_prefill_logits.pkl',
                submission_generator.prefill,
                dumping_range='qlv4_linear',
                dumping_mode='only-in-out', 
                qlv4_skip_output_rounding=False,
                dumping_before_rounding=True,
            )
    model_compressor.set_model_to_dump_golden_model(
                './ci_test_file/submission_decode_logits.pkl',
                submission_generator.decode,
                dumping_range='qlv4_linear',
                dumping_mode='only-in-out', 
                qlv4_skip_output_rounding=False,
                dumping_before_rounding=True,
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
        output_batch_submission = submission_generator.generate(**input_batch, max_length = seq_len+3, min_new_tokens = 10)
        #output_batch_submission = submission_generator.generate(**input_batch, max_length=2048, **gen_kwargs)


    is_logit_same(golden_model_file_path='./ci_test_file/golden_prefill_logits.pkl',
                  comparison_model_file_path = './ci_test_file/submission_prefill_logits.pkl', 
                  mcm_name_to_check='lm_logits')
    
    is_logit_same(golden_model_file_path='./ci_test_file/golden_decode_logits.pkl',
                  comparison_model_file_path ='./ci_test_file/submission_decode_logits.pkl',
                  mcm_name_to_check='lm_logits',
                  decode = True)
    
    print("gptj forward ci test is passed")
    
    

if __name__ == "__main__":
    compare_model_outputs()
