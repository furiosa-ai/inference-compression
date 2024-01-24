import yaml
import os 
from torch.utils.data import DataLoader
from transformers.utils.fx import symbolic_trace
import model_compressor


gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
}

def make_dataloader(data_object, batch_size):
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx]})
    
    return DataLoader(data_list, batch_size)


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(model, data_object, model_script_path):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    calib_dataloader = make_dataloader(data_object, model_script['calib_batch_size'])
    
    model = symbolic_trace(model, input_names=["input_ids", "attention_mask"])
    
    quant_model = model_compressor.create_quantsim_model(
        model,
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=calib_dataloader,
    )

    model_compressor.save_qformat(
        model,
        qformat_out_path="./qformat.yaml",
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
    )
