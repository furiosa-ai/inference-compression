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


def get_quant_model(sut, model_source, model_script_path, n_calib, recalibrate):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    qlevel = model_script["qlevel"]
    
    sut.model.config.use_cache = False
    
    output_path='./quantization/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    qformat_path = f"{output_path}/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    qparam_path = f"{output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    
    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
    else:
        calib_dataloader = make_dataloader(sut.qsl, model_script['calib_batch_size'], n_calib)

        if model_source == 'mlperf_submission':
            from .calib_dataloader import make_packed_calib_data_loader
            calib_dataloader = make_packed_calib_data_loader(calib_dataloader, 512, 0)
    
    
    
    
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
    
    
    
    return quant_model
    