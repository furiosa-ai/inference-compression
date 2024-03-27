import yaml
import os 
from torch.utils.data import DataLoader
from .custom_symbolic_trace import custom_symbolic_trace
import model_compressor
import torch
from .calib_dataloader import make_dataloader
#calib_dataset_path
from transformers.generation.utils import *
from utils import make_gm_code

import torch._decomp
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx


def std_decompositions():
    return {
        **torch._decomp.core_aten_decompositions(),
        **torch._decomp.get_decompositions(
            [
                torch.ops.aten.addmm,
                torch.ops.aten.gelu,
                torch.ops.aten.native_layer_norm,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
                torch.ops.aten.embedding,
                torch.ops.aten._unsafe_view,
                torch.ops.aten.upsample_nearest2d,
                torch.ops.aten.clamp_min,
                torch.ops.aten.clamp_max,
                torch.ops.aten.relu_,
                torch.ops.aten.roll,
                torch.ops.aten.linalg_vector_norm,
                torch.ops.aten._native_batch_norm_legit,
                torch.ops.aten._native_batch_norm_legit_no_training,
                torch.ops.aten.relu,
                torch.ops.aten.clamp,
                torch.ops.aten.repeat,
            ]
        ),
    }



def load_model_script(model_script_path):
    with open(model_script_path, 'r') as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(sut, model_script_path, n_calib, recalibrate):
    #Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    qlevel = model_script["qlevel"]

    output_path='./quantization/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    qformat_path = f"{output_path}/qformat_{model_script_path.split('.')[1].split('/')[-1]}.yaml" 
    qparam_path = f"{output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    model, input_names, concrete_args = custom_symbolic_trace(sut.model)

    if os.path.exists(qformat_path) and os.path.exists(qparam_path) and recalibrate == False:
        calib_dataloader = None
        org_model = None
    else:
        calib_dataloader = make_dataloader(sut.qsl, model_script['calib_batch_size'], n_calib)
        import copy
        org_model = copy.deepcopy(model) if qlevel >=3 else None

    model.config.use_cache = False

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
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=calib_dataloader,
        disable_inout=(True,True),
        )
    
    #if False:
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
            kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
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
        kv_dtype = model_script["kv_dtype"] if "kv_dtype" in model_script else 'bf16',
        qlevel=model_script["qlevel"],
        target_machine=model_script["target_machine"],
        dataloader=None,
        disable_inout=(True,True),
        )
    

    dummy_batch = next(iter(calib_dataloader))
    for key in dummy_batch:
        dummy_batch[key] = dummy_batch[key].cuda()

    with torch.no_grad():
        quant_model(**dummy_batch)

    quant_model.eval()
    
    gm = functionalize(quant_model, remove='mutations_and_views')
    gm = make_fx(
        gm,
        tracing_mode='real',
        _allow_non_fake_inputs=True,
        decomposition_table=std_decompositions(),
    )(
        dummy_batch["input_ids"],
        dummy_batch["attention_mask"],
        dummy_batch["token_type_ids"],
    )

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    
    gm = torch.compile(quant_model)
    output = torch._dynamo.explain(
        gm,
        dummy_batch["input_ids"],
        dummy_batch["attention_mask"],
        dummy_batch["token_type_ids"],
        
    )
    with open("code_v1.1furiosa.txt", 'w') as fp: fp.write(gm.code) 
    return quant_model