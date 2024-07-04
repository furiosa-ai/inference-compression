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
from generator_RNGD import (MLPerfSubmissionBeamSearch,
                            expand_inputs_for_generation)
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer
from transformers.utils.fx import get_concrete_args
import pickle


EARYLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 50256
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 30
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NUM_RETURN_SEQUENCES = 1
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.int8
BUCKET_SIZE = 2048
NUM_REAL_BATCH = 1


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
    # model_config = AutoConfig.from_pretrained(model_path)
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
    
    
           
    # from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
    from backend_RNGD import GPTJForCausalLM 
    submission_model = GPTJForCausalLM.from_pretrained(model_path, config=model_config).to(device)
    
    
    submission_generator = quantization.get_quant_model(model = submission_model, 
                                                     calib_dataset_path = calib_dataset_path, 
                                                     model_script_path = model_script_path, 
                                                     calib_without_padding = False, 
                                                     recalibrate = False, 
                                                     qformat_path = qformat_path, 
                                                     qparam_path = qparam_path,
                                                     immigrate_qparams = True)
    
    
    #Turn on dumping 
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
            './ci_test_file/submission_prefill_logits_rngd.pkl',
            submission_generator.prefill,
            dumping_range='lm_head',
            dumping_mode='only-in-out', 
            qlv4_skip_output_rounding=False,
            dumping_before_rounding=True,
            dump_in_append_mode=True)
            
    model_compressor.set_model_to_dump_golden_model(
        './ci_test_file/submission_decode_logits_rngd.pkl',
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
        #output_batch_submission = submission_generator.generate(**input_batch, max_length = seq_len+3, min_new_tokens = 10)
        #output_batch_submission = submission_generator.generate(**input_batch, max_length=2048, **gen_kwargs)

        logits_processor = LOGITS_PROCESSOR(
                input_batch['input_ids'].shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
            )
        # stopping_criteria = STOPPING_CRITERIA(
        #         MAX_LENGTH,
        #         getattr(submission_generator.model_config, "max_position_embeddings", None),
        #     )
            #The stopping_criteria cannot be used for MLPerf BeamSearch, as the length of every input_ids is fixed to max_prompt_length
        stopping_criteria = None


        beam_scorer = BeamSearchScorer(
            batch_size=input_batch['input_ids'].shape[0],
            num_beams=NUM_BEAMS,
            device=input_batch['input_ids'].device,
            length_penalty=LENGTH_PENALTY,
            do_early_stopping=EARYLY_STOPPING,
            num_beam_hyps_to_keep=NUM_RETURN_SEQUENCES,
            max_length=MAX_LENGTH,
        )
        input_ids_tensor, input_masks_tensor_dict = expand_inputs_for_generation(
            input_ids=input_batch['input_ids'],
            expand_size=NUM_BEAMS,
            attention_mask= input_batch['attention_mask'],
        )
        input_masks_tensor = input_masks_tensor_dict["attention_mask"]

        output_batch = submission_generator.generate(
            input_ids=input_ids_tensor,
            attention_mask=input_masks_tensor,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=MAX_LENGTH,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            return_dict_in_generate=RETURN_DICT_IN_GENERATE,
            kv_dtype=KV_DTYPE,
            bucket_size=BUCKET_SIZE,
        )



    check_logits(golden_model_file_path='./ci_test_file/golden_prefill_logits.pkl',
                    comparison_model_file_path ='./ci_test_file/submission_prefill_logits_rngd.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = False)

    check_logits(golden_model_file_path='./ci_test_file/golden_decode_logits.pkl',
                    comparison_model_file_path ='./ci_test_file/submission_decode_logits_rngd.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = True)
    
    print("gptj forward ci test is passed")
    
    

if __name__ == "__main__":
    compare_model_outputs()
