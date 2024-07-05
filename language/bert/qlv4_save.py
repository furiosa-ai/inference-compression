# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import subprocess
import sys
import torch

import mlperf_loadgen as lg

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "lon"))
import model_compressor
from absl import app, flags

from quantization import get_quant_model
from utils import random_seed, set_optimization


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["tf", "pytorch", "onnxruntime", "tf_estimator", "ray"],
        default="tf",
        help="Backend",
    )
    parser.add_argument(
        "--scenario",
        choices=["SingleStream", "Offline", "Server", "MultiStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="use quantized model (only valid for onnxruntime backend)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable profiling (only valid for onnxruntime backend)",
    )
    parser.add_argument(
        "--mlperf_conf", default="build/mlperf.conf", help="mlperf rules config"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--audit_conf",
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="Maximum number of examples to consider (not limited by default)",
    )
    parser.add_argument(
        "--network",
        choices=["sut", "lon", None],
        default=None,
        help="Loadgen network mode",
    )
    parser.add_argument("--node", type=str, default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--sut_server",
        nargs="*",
        default=["http://localhost:8000"],
        help="Address of the server(s) under test.",
    )
    parser.add_argument(
        "--model_script_path",
        default="./quantization/model_script/Qlevel4_RGDA0-W8A8KV8-PTQ.yaml",
        help="",
    )
    parser.add_argument(
        "--use_mcp", action="store_true", help="use mcp to quantize the model"
    )
    parser.add_argument("--n_calib", type=int, default=-1)
    parser.add_argument(
        "--torch_optim",
        default="none",
        type=str,
        choices=["default", "none"],
        help="Torch optimization.",
    )
    parser.add_argument(
        "--n_layers",
        default="-1",
        type=int,
        help="set the number of layers.",
    )
    parser.add_argument(
        "--model_source",
        default="mlperf_submission",
        type=str,
        choices=[
            "huggingface_rngd_gelu",
            "mlperf_submission",
            "experimental_huggingface_unsplit_packed",
        ],
        help="choose model source",
    )
    parser.add_argument(
        "--output_path",
        default='./quantization/output',
        help="",
    )

    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream,
}


def qlv4_save():
    # ---------------------------------------------------------
    # Setting for ci test
    # ---------------------------------------------------------
    args = get_args()
    set_optimization(args)
    random_seed()

    sut = None
    args.backend = "pytorch"
    args.max_examples = 1
    args.recalibrate = True
    args.use_mcp = True
    args.accuracy = True
    args.torch_optim = "none"
    args.model_script_path = (
        "./quantization/model_script/Qlevel4_RGDA0-W8A8KV8-PTQ.yaml"
    )

    from pytorch_SUT import get_pytorch_sut

    # ---------------------------------------------------------
    # get model
    # ---------------------------------------------------------
    sut = get_pytorch_sut(args)
    sut.model = get_quant_model(
        sut,
        args.model_source,
        args.model_script_path,
        args.n_calib,
        False,
        output_path=args.output_path,
    )

    if args.model_source =="mlperf_submission":
        model = sut.model.model
    else:
        model= sut.model
    
    torch.save(model.state_dict(), args.output_path + '/qlv4.bin')
    print("qlv4 model is saved well")


if __name__ == "__main__":
    qlv4_save()
