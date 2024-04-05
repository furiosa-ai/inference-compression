import yaml
import os
import torch
from torch.utils.data import DataLoader
from transformers.utils.fx import symbolic_trace
import model_compressor
import furiosa_llm_models
from typing import Optional 
from dataset import Dataset
import copy




gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    # only beam_size 4 is allowed for official submission
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")),
}


GENERATOR_DICT = {
    furiosa_llm_models.gptj.paged_attention_concat.GPTJForCausalLM : furiosa_llm_models.generators.paged_attention_generator_concat.QuantPagedAttentionGenerator,
    furiosa_llm_models.gptj.paged_attention_concat_rope.GPTJForCausalLM : furiosa_llm_models.generators.paged_attention_generator_concat.QuantPagedAttentionGenerator,
}

def get_total_block_space(config, num_blocks = 32 , block_size = 16, bucket_size = 512):
    #artibrary set to accomodate input prompt & generated summary
    num_blocks = (1633+100)*2+1
    block_size = 1 
    bucket_size = 2048
    example_block_per_layer_shape = (
        num_blocks,
        block_size,
        config.n_head,
        int(config.n_embd / config.n_head),    )

    total_block_space = []
    for _ in range(0, config.n_layer):
        total_block_space.append(
            (
                torch.zeros(example_block_per_layer_shape),  # key
                torch.zeros(example_block_per_layer_shape),  # value
            )
        )
    return (bucket_size, total_block_space)


def make_calib_dataloader(calib_dataset_path, batch_size):
    data_object = Dataset(calib_dataset_path, batch_size)
    data_list = []
    for idx in range(len(data_object.source_encoded_input_ids)):
        data_list.append({'input_ids': data_object.source_encoded_input_ids[idx], 'attention_mask': data_object.source_encoded_attn_masks[idx], 'position_ids': torch.arange(
                len(data_object.source_encoded_input_ids[idx][0]))})
    return DataLoader(data_list, batch_size)


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_autoscale_calib_config(model_script, model, calib_dataloader):
    from .autoscale import extract_kwargs 
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    args = dotdict(model_script)
    autoscale_calib_cfg = extract_kwargs.get_autoscale_calib_cfg(args, model, calib_dataloader)
    return autoscale_calib_cfg



def get_quant_model(model, calib_dataset_path, model_script_path, recalibrate):
    # Load model script and calibration dataloader (Refer to inference-compression/language/gpt-j/README.md on how to download evaluation and calibration dataset )
    model_script = load_model_script(model_script_path)

    qformat_path = f"./quantization/output/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    qparam_path = f"./quantization/output/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"
    
    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
    else:
        calib_dataloader = make_calib_dataloader(calib_dataset_path, model_script['calib_batch_size'])
  
    run_autoscale = model_script.get("autoscale", 'disabled') != 'disabled'  
     #prepare for autoscale 
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(model_script, model, calib_dataloader)


    model_type = type(model)

    if calib_dataloader:
        prefill_model_for_calib, _, _ = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = True)
        # Extract necessary parameters to initialize QuantPreTrainedModel
        prefill_model_for_calib = model_compressor.create_quantsim_model(
            prefill_model_for_calib,
            qformat_path = None,
            qparam_path = None,
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
            act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
            dataloader=calib_dataloader,
            disable_inout=(True, True),
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        )

        model_compressor.calibrate(
            model=prefill_model_for_calib,
            model_name=model_script["model"],
            calib_dataloader=calib_dataloader,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            percentile=model_script["percentile"],
            target_machine=model_script["target_machine"],
            act_zp_equalizing=model_script["act_zp_equalizing"] if  "act_zp_equalizing" in model_script else 'disabled',
            autoscale=model_script["autoscale"] if run_autoscale else "disabled",
            autoscale_calib_method=(model_script["autoscale_calib_method"] if run_autoscale else 'auto'),
            autoscale_calib_kwargs=autoscale_calib_cfg if run_autoscale else None,
        )

        model_compressor.save(
                prefill_model_for_calib,
                qparam_out_path=qparam_path,
                qformat_out_path=qformat_path,
                weight_calib_method=model_script["weight_calib_method"],
                weight_granularity=model_script["weight_granularity"],
                weight_dtype=model_script["weight_dtype"],
                weight_nbits=model_script["weight_nbits"],
                act_calib_method=model_script["act_calib_method"],
                act_granularity=model_script["act_granularity"],
                act_dtype=model_script["act_dtype"],
                act_nbits=model_script["act_nbits"],
                #disable_mods=args.disable_quant_list,
            )


        del prefill_model_for_calib


    prefill_model, prefill_input_names, prefill_concrete_args = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = True)
    decode_model, decode_input_names, decode_concrete_args = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = False)
    
    input_names = {
        "prefill_input_names" : prefill_input_names,
        "decode_input_names" : decode_input_names,
    }

    concrete_args = {
        "prefill_concrete_args": prefill_concrete_args,
        "decode_concrete_args": decode_concrete_args,
    }

    prefill_model = model_compressor.create_quantsim_model(
        prefill_model,
        qformat_path = qformat_path,
        qparam_path = qparam_path,
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
        act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
        dataloader=None,
        disable_inout=(True, True),
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        decode_phase = False,
        model_name = "GPTJForCausalLM",
    )

    decode_model = model_compressor.create_quantsim_model(
        decode_model,
        qformat_path = qformat_path,
        qparam_path = qparam_path,
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
        act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
        dataloader=None,
        disable_inout=(True, True),
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        decode_phase = True,
        model_name = "GPTJForCausalLM",
    )

    quant_models = {
        "prefill_model": prefill_model,
        "decode_model": decode_model,
    }

    quant_causallm = model_compressor.helper.QuantCausalLM(quant_models, model_type, input_names, concrete_args)
    
    if model_type in GENERATOR_DICT.keys():
        bucket_size, total_block_space = get_total_block_space(prefill_model.config)
        return GENERATOR_DICT[model_type](quant_causallm, total_block_space, bucket_size)
    else: 
        return model_compressor.helper.QuantCausalLM(quant_models, model_type, input_names, concrete_args)