import yaml
import os
from torch.utils.data import DataLoader
from .custom_symbolic_trace import custom_symbolic_trace
import model_compressor
import torch
from .calib_dataloader import make_dataloader

# calib_dataset_path
from transformers.generation.utils import *


def load_model_script(model_script_path):
    with open(model_script_path, "r") as f:
        model_script = yaml.safe_load(f)

    return model_script


def get_quant_model(
    sut,
    model_script_path,
    calib_source,
    qformat_path=None,
    qparam_path=None,
    n_calib=-1,
):
    # Load model script and calibration dataloader
    model_script = load_model_script(model_script_path)
    
    qlevel = model_script["qlevel"]

    model, input_names, concrete_args = custom_symbolic_trace(sut.model)

    # 실험 시 매번 새로 수행하기 위해 주석처리
    # if os.path.exists(qformat_path) and os.path.exists(qparam_path):
    #     calib_dataloader = None
    #     org_model = None
    # else:
    #     calib_dataloader = make_dataloader(sut.qsl, model_script['calib_batch_size'], calib_source,n_calib)
    #     import copy
    #     org_model = copy.deepcopy(model) if qlevel >=3 else None

    calib_dataloader = make_dataloader(
        sut.qsl, model_script["calib_batch_size"], calib_source, n_calib
    )
    org_model = None  # BERT-SMQ는 Qlevel1에서 테스트 수행함.
    run_autoscale = model_script.get("autoscale", "disabled") != "disabled"
    if run_autoscale:
        from .autoscale.extract_kwargs import get_autoscale_calib_cfg

        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        args = dotdict(model_script)
        _autuscale_calib_cfg = get_autoscale_calib_cfg(
            args, sut.model, calib_dataloader
        )

    # origin_model = model
    quant_model = model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path if calib_dataloader is None else None,
        qparam_path=qparam_path if calib_dataloader is None else None,
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
        act_zp_equalizing=(
            model_script["act_zp_equalizing"] if run_autoscale else "disabled"
        ),
        dataloader=calib_dataloader,
        disable_inout=(True, True),
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
            act_zp_equalizing=(
                model_script["act_zp_equalizing"] if run_autoscale else "disabled"
            ),
            autoscale=model_script["autoscale"] if run_autoscale else "disabled",
            autoscale_calib_method=(
                model_script["autoscale_calib_method"] if run_autoscale else "auto"
            ),
            autoscale_calib_kwargs=_autuscale_calib_cfg if run_autoscale else None,
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
            disable_inout=(True, True),
        )

    return quant_model
