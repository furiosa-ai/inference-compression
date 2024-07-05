import yaml
import os
import torch
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


FURIOSA_GENERATOR_DICT = {
    furiosa_llm_models.gptj.symbolic.mlperf_submission.GPTJForCausalLM : furiosa_llm_models.generators.paged_attention_optimized_generator_beam_search_optimized.PagedAttentionGeneratorBeamSearch,
    # furiosa_llm_models.gptj.symbolic.preallocated_concat_rope.GPTJForCausalLM : furiosa_llm_models.generators.symbolic.quant_preallocated_concat_generator.QuantPreAllocatedConcatGenerator,
    furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM: furiosa_llm_models.generators.symbolic.quant_paged_attention_generator.QuantPagedAttentionGenerator,
    # furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope.GPTJForCausalLM: furiosa_llm_models.generators.paged_attention_optimized_generator_beam_search.PagedAttentionGeneratorBeamSearch,
    # furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope_erf_gelu.GPTJForCausalLM: furiosa_llm_models.generators.paged_attention_optimized_generator_beam_search.PagedAttentionGeneratorBeamSearch,
    
}

def get_total_block_space(config, batch_size = 1, block_size = 1, bucket_size = 2048, kv_dtype = 'float32'):
    #artibrary set to accomodate input prompt & generated summary
    
    if kv_dtype == 'float32':
        block_dtype = torch.float32
    elif kv_dtype == 'bf16':
        block_dtype = torch.bfloat16
    elif kv_dtype == 'int8':
        block_dtype = torch.int8
    else:
        raise NotImplementedError
    
    num_blocks = batch_size * 4  * 2 * (bucket_size) * block_size + 1
    example_block_per_layer_shape = (
        num_blocks,
        block_size,
        config.n_head,
        int(config.n_embd / config.n_head),    )

    total_block_space = []
    for _ in range(0, config.n_layer):
        total_block_space.append(
            (
                torch.zeros(example_block_per_layer_shape, dtype = block_dtype),  # key
                torch.zeros(example_block_per_layer_shape, dtype = block_dtype),  # value
            )
        )
    return (bucket_size, total_block_space)



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



def get_quant_model(model, model_script_path, recalibrate, calib_dataset_path=None, calib_without_padding=False, qformat_path = None, qparam_path = None, immigrate_qparams = False):
    # Load model script and calibration dataloader (Refer to inference-compression/language/gpt-j/README.md on how to download evaluation and calibration dataset )
    model_script = load_model_script(model_script_path)
    
    if qformat_path is None and qparam_path is None:
        qformat_path = f"./quantization/output/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
        qparam_path = f"./quantization/output/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"
    
    
    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
    else:
        if type(model) == furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM:
            from .calibration_utils.paged_attention_utils import make_calib_dataloader_for_paged_attention
            bucket_size, total_block_space = get_total_block_space(model.config, kv_dtype = 'float32') #kv_dtype are set as float32 to enable dummy forwarding before calibration.
            calib_dataloader =  make_calib_dataloader_for_paged_attention(calib_dataset_path, model_script['calib_batch_size'], bucket_size, total_block_space)
        elif type(model) in [furiosa_llm_models.gptj.symbolic.mlperf_submission.GPTJForCausalLM,]:
            from .calibration_utils.paged_attention_optimized_packed_utils import make_calib_dataloader_for_paged_attention_packed
            bucket_size, total_block_space = get_total_block_space(model.config, kv_dtype = 'float32') #kv_dtype are set as float32 to enable dummy forwarding before calibration.
            calib_dataloader =  make_calib_dataloader_for_paged_attention_packed(calib_dataset_path, model.config, model_script['calib_batch_size'], bucket_size, total_block_space)
        # elif type(model) == furiosa_llm_models.gptj.paged_attentin_rope
        
        else:
            from .calibration_utils.make_calib_dataloader import make_calib_dataloader
            calib_dataloader = make_calib_dataloader(calib_dataset_path, model_script['calib_batch_size'], calib_without_padding)
            
  
    run_autoscale = model_script.get("autoscale", 'disabled') != 'disabled'  
     #prepare for autoscale 
    if run_autoscale:
        autoscale_calib_cfg = get_autoscale_calib_config(model_script, model, calib_dataloader)


    model_type = type(model)

    if calib_dataloader:
        if type(model) == furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM:
        #The following usage of mcp helper tracer will be removed soon once paged_attention_rope_rngd_gelu is added   
            model_for_calib, _, _ = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = False, disable_check=True)    
        else:
            model_for_calib = model.trace_prefill()

        # Extract necessary parameters to initialize QuantPreTrainedModel
        model_for_calib = model_compressor.create_quantsim_model(
            model_for_calib,
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
            disable_inout=(True, False),
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        )

        model_compressor.calibrate(
            model=model_for_calib,
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
                model_for_calib,
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
                kv_dtype=model_script["kv_dtype"] if  "kv_dtype" in model_script else 'bf16',
                disable_inout=(True, False),
            )


        del model_for_calib

    if model_type == furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM:
        #There is no prefill model for paged_attention_rope as past_key_values are always fed to the model
        #The following usage of mcp helper tracer will be removed soon once paged_attention_rope_rngd_gelu is added   
        decode_model, decode_input_names, decode_concrete_args = model_compressor.helper.gptj_custom_symbolic_trace(model, prefill_mode = False)
        decode_model = model_compressor.create_quantsim_model(
            decode_model,
            qformat_path = qformat_path,
            qparam_path = qparam_path,
            qlevel=model_script["qlevel"],
            target_machine=model_script["target_machine"],
            delete_org_weight=True,
        )
        generator = FURIOSA_GENERATOR_DICT[model_type]

        # only a single graph is  required for paged_attention_rope 
        bucket_size, total_block_space = get_total_block_space(decode_model.config, kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16')
        return generator(decode_model, model_type, total_block_space, bucket_size) 
    
    else: 
        traced_models = model.trace_all()
        
        input_names = {
        "prefill_input_names": traced_models["prefill"].input_names,
        "decode_input_names": traced_models["decode"].input_names,
        }
        concrete_args = {
        "prefill_concrete_args": traced_models["prefill"].concrete_args,
        "decode_concrete_args": traced_models["decode"].concrete_args,
        }
        
        prefill_model = model_compressor.create_quantsim_model(
            traced_models["prefill"],
            qformat_path = qformat_path,
            qparam_path = qparam_path,
            qlevel=model_script["qlevel"],
            target_machine=model_script["target_machine"],
            # decode_phase = True,
            delete_org_weight=True,
            immigrate_qparams = immigrate_qparams,
        )

        decode_model = model_compressor.create_quantsim_model(
            traced_models["decode"],
            qformat_path = qformat_path,
            qparam_path = qparam_path,
            qlevel=model_script["qlevel"],
            target_machine=model_script["target_machine"],
            decode_phase = True,
            delete_org_weight=True,
            quantized_prefill_model=prefill_model,
            immigrate_qparams = immigrate_qparams,
        )

        quant_models = {
            "prefill_model": prefill_model,
            "decode_model": decode_model,
        }



        if model_type in FURIOSA_GENERATOR_DICT.keys():
            generator = FURIOSA_GENERATOR_DICT[model_type]
            # if generator == furiosa_llm_models.generators.symbolic.quant_preallocated_concat_generator.QuantPreAllocatedConcatGenerator:
            #     return generator(quant_causallm, bucket_size = 2048)
            if generator == furiosa_llm_models.generators.paged_attention_optimized_generator_beam_search_optimized.PagedAttentionGeneratorBeamSearch:
                quant_models["prefill_model"].concrete_args = concrete_args["prefill_concrete_args"]
                quant_models["decode_model"].concrete_args = concrete_args["decode_concrete_args"]
                return generator(prefill=quant_models["prefill_model"], decode=quant_models["decode_model"], kv_dtype=torch.int8)
            else:
                
                quant_causallm = model_compressor.helper.QuantCausalLM(quant_models, model_type)
                bucket_size, total_block_space = get_total_block_space(prefill_model.config, kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16')
                return generator(quant_causallm, total_block_space, bucket_size)
        else: 
            return model_compressor.helper.QuantCausalLM(quant_models, model_type, input_names, concrete_args)