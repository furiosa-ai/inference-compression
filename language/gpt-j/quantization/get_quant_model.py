import yaml
import os 
import torch
from torch.utils.data import DataLoader
from transformers.utils.fx import symbolic_trace
import model_compressor
from typing import Optional
from .QuantGenerationModel import QuantPreTrainedModel


gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
}

def make_dataloader(data_object, batch_size):
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(len(data_object.source_encoded_input_ids[idx][0]))})
    
    return DataLoader(data_list, batch_size)


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script

        
    


def get_quant_model(model, data_object, model_script_path):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    calib_dataloader = make_dataloader(data_object, model_script['calib_batch_size'])
    
    #Extract necessary parameters to initialize QuantPreTrainedModel
    model_config = model.config
    model_type = type(model)


    model = symbolic_trace(model, input_names=["input_ids", "position_ids","attention_mask"])

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
    
    # model_compressor.calibrate(
    #         model=model,
    #         #model_name=model_name,
    #         weight_calib_method=model_script["weight_calib_method"],
    #         act_calib_method=model_script["act_calib_method"],
    #         outlier_calib_cfg=model_script['outlier_compensation'],
    #         group_size=args.group_size,
    #         percentile=args.percentile,
    #         is_dynamic_quant=args.is_dynamic_quant,
    #         split_mode=args.split_mode,
    #         autoscale=args.autoscale,
    #         autoscale_calib_method=args.autoscale_calib_method,
    #         autoscale_calib_kwargs=calib_cfg['autoscale'],
    #         autoclip=args.autoclip,
    #         target_machine=args.target_machine,
    #         calib_dataloader=loader_calib,
    #         data_preprocessor=explicit_preproc_fn,
    # )

    # model_compressor.save_qformat(
    #     model,
    #     qformat_out_path="./qformat.yaml",
    #     weight_calib_method=model_script["weight_calib_method"],
    #     weight_granularity=model_script["weight_granularity"],
    #     weight_dtype=model_script["weight_dtype"],
    #     weight_nbits=model_script["weight_nbits"],
    #     act_calib_method=model_script["act_calib_method"],
    #     act_granularity=model_script["act_granularity"],
    #     act_dtype=model_script["act_dtype"],
    #     act_nbits=model_script["act_nbits"],
    # )
    
    
    
    quant_model.recompile()
    
    
    return QuantPreTrainedModel(quant_model, model_config, model_type)
 
