import yaml
import os 
from torch.utils.data import DataLoader
from .custom_symbolic_trace import custom_symbolic_trace
import model_compressor
import torch
from .calib_dataloader import make_dataloader
#calib_dataset_path
from transformers.generation.utils import *


def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(sut, model_script_path, calib_source, qformat_path=None, qparam_path=None, n_calib=-1):
    #Load model script and calibration dataloader
    
    model_script = load_model_script(model_script_path)
    
    qlevel = model_script["qlevel"]

    model, input_names, concrete_args = custom_symbolic_trace(sut.model)

    if os.path.exists(qformat_path) and os.path.exists(qparam_path):
        calib_dataloader = None
        org_model = None
    else:
        calib_dataloader = make_dataloader(sut.qsl, model_script['calib_batch_size'], calib_source,n_calib)
        import copy
        org_model = copy.deepcopy(model) if qlevel >=3 else None

    #origin_model = model
    quant_model = model_compressor.create_quantsim_model(
        model,
        qformat_path = qformat_path if calib_dataloader is None else None,
        qparam_path = qparam_path if calib_dataloader is None else None,
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
        disable_inout=(True,True),
        )
    

    if calib_dataloader:

        model_compressor.calibrate(
            quant_model,
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
        )

        model_compressor.save(
            quant_model,
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
        )

        quant_model.recompile()

    if org_model:
        quant_model = model_compressor.create_quantsim_model(
        org_model,
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
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=None,
        disable_inout=(True,True),
        )
    


    return quant_model