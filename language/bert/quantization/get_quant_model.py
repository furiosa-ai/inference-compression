import yaml
import os 
from torch.utils.data import DataLoader
from .calib_dataloader import make_dataloader
import model_compressor
import torch

#calib_dataset_path
from transformers.generation.utils import *


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(sut, model_source, model_script_path, n_calib, recalibrate, use_packed_dataloader=True, output_path='./quantization/output', qformat_path = None, qparam_path=None):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    qlevel = model_script["qlevel"]
    
    sut.model.config.use_cache = False   
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if qformat_path is None:
        qformat_path = f"{output_path}/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    
    if qparam_path is None:
        qparam_path = f"{output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    
    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
    else:
        from .calib_dataloader import load_bert_calibration_data
        calib_eval_features = load_bert_calibration_data(sut.qsl, n_calib)
        
        if model_source == 'mlperf_submission':
            from .calib_dataloader import make_packed_calib_data_loader
            if use_packed_dataloader:
                calib_dataloader = make_packed_calib_data_loader(calib_eval_features, model_script['calib_batch_size'], n_calib, pad_token_id=0, bucket_size=384, compact_mask=False) 
            else:
                calib_dataloader = make_dataloader(calib_eval_features, model_script['calib_batch_size'], n_calib, include_position_ids=True)
        elif model_source == 'experimental_huggingface_unsplit_packed':
            from .calib_dataloader import make_packed_calib_data_loader
            calib_dataloader = make_packed_calib_data_loader(calib_eval_features, model_script['calib_batch_size'], n_calib, pad_token_id=0, bucket_size=384, compact_mask=True)                 
        else:            
            calib_dataloader = make_dataloader(calib_eval_features, model_script['calib_batch_size'], n_calib)
           

    if calib_dataloader:
        #model_for_calib = symbolic_trace(sut.model, input_names=input_names, disable_check=False)
        model_for_calib = sut.model.trace()
        
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
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
            act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
            qlevel=model_script["qlevel"],
            target_machine=model_script["target_machine"],
            dataloader=calib_dataloader,
            disable_inout=(True,True),
        )
    
        model_compressor.calibrate(
            model_for_calib,
            calib_dataloader=calib_dataloader,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
            act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
            percentile=model_script["percentile"],
            target_machine=model_script["target_machine"],
        )

        model_compressor.save(
            model_for_calib,
            qformat_out_path=qformat_path,
            qparam_out_path=qparam_path,
            weight_calib_method=model_script["weight_calib_method"],
            weight_granularity=model_script["weight_granularity"],
            weight_dtype=model_script["weight_dtype"],
            weight_nbits=model_script["weight_nbits"],
            act_calib_method=model_script["act_calib_method"],
            act_granularity=model_script["act_granularity"],
            act_dtype=model_script["act_dtype"],
            act_nbits=model_script["act_nbits"],
            kv_dtype=model_script["kv_dtype"] if  "kv_dtype" in model_script else 'bf16',
            disable_inout=(True, True),
        )
        
        del model_for_calib
    

    #model = symbolic_trace(sut.model, input_names=input_names,disable_check=False)
    model = sut.model.trace()

    quant_model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        weight_calib_method=model_script["weight_calib_method"],
        weight_granularity=model_script["weight_granularity"],
        weight_dtype=model_script["weight_dtype"],
        weight_nbits=model_script["weight_nbits"],
        act_calib_method=model_script["act_calib_method"],
        act_granularity=model_script["act_granularity"],
        act_dtype=model_script["act_dtype"],
        act_nbits=model_script["act_nbits"],
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        qlevel=model_script["qlevel"],
        act_zp_equalizing=(model_script["act_zp_equalizing"] if model_script["act_zp_equalizing"] else 'disabled'),
        target_machine=model_script["target_machine"],
        dataloader=None,
        disable_inout=(True,True),
    )
    
    if model_source == 'mlperf_submission':
        from furiosa_llm_models.generators.bert_generator import BertUnsplitPackedGenerator
        generator = BertUnsplitPackedGenerator(model=quant_model, compact_mask=False)
    elif model_source == 'experimental_huggingface_unsplit_packed':
        from furiosa_llm_models.generators.bert_generator import BertUnsplitPackedGenerator
        generator = BertUnsplitPackedGenerator(model=quant_model, compact_mask=True)   
    else:
        generator = None
    
    if generator is None:
        return quant_model
    
    return generator
    