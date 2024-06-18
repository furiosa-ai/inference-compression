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
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="load already existing quantization metadata",
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
        default="unsplit_packed",
        type=str,
        choices=["huggingface_rngd_gelu", "mlperf_submission"],
        help="choose model source",
    )

    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream,
}


def dump_using_mlperf_loadgens(args, sut, dumpfile_path):
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "bert", args.scenario)
    settings.FromConfig(args.user_conf, "bert", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = os.environ.get("LOG_PATH")
    if not log_path:
        log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    print("Running LoadGen test...")

    model_compressor.set_model_to_dump_golden_model(
        dumpfile_path,
        sut.model,
        dumping_range="qlv4_linear",
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
    )

    lg.StartTestWithLogSettings(
        sut.sut, sut.qsl.qsl, settings, log_settings, args.audit_conf
    )
    if args.accuracy and not os.environ.get("SKIP_VERIFY_ACCURACY"):
        cmd = "python3 {:}/accuracy-squad.py {}".format(
            os.path.dirname(os.path.abspath(__file__)),
            "--max_examples {}".format(args.max_examples) if args.max_examples else "",
        )
        subprocess.check_call(cmd, shell=True)

    print("Done!")

    if sut:
        print("Destroying SUT...")
        lg.DestroySUT(sut.sut)

        print("Destroying QSL...")
        lg.DestroyQSL(sut.qsl.qsl)


def cmp_qparam(model_script_path, hf_output_path, mlperf_output_path):
    import numpy as np

    hf_qparam_path = (
        f"{hf_output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"
    )
    mlperf_qparam_path = f"{mlperf_output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    hf_qparam = np.load(hf_qparam_path, allow_pickle=True).item()
    mlperf_qparam = np.load(mlperf_qparam_path, allow_pickle=True).item()

    for module_name, module_qparam in mlperf_qparam.items():
        try:
            hf_data = hf_qparam[module_name]
        except:
            continue

        for qparam_name, qparam in hf_data.items():
            if qparam is None:
                continue

            if not np.array_equal(module_qparam[qparam_name], qparam):
                print(
                    "Failed ",
                    module_name,
                    qparam_name,
                    (abs(module_qparam[qparam_name] - qparam) / abs(qparam)).max(),
                )
            else:
                print("Passed ", module_name, qparam_name)


def main():
    args = get_args()

    set_optimization(args)
    random_seed()

    sut = None
    args.backend = "pytorch"
    args.max_examples = 1
    args.recalibrate = True
    args.use_mcp = True
    args.model_script_path = (
        "./quantization/model_script/Qlevel4_RGDA0-W8A8KV8-PTQ.yaml"
    )
    args.accuracy = True
    args.torch_optim = "none"

    from pytorch_SUT import get_pytorch_sut

    golden_file_path = "./dump-ci-test/hf_rngd_gelu.pkl"
    comparison_file_path = "./dump-ci-test/mlperf.pkl"

    # Dump golden model
    args.model_source = "huggingface_rngd_gelu"
    golden_output_path = "./test/output_rngd_gelu/hf"
    sut = get_pytorch_sut(args)
    sut.model = get_quant_model(
        sut,
        args.model_source,
        args.model_script_path,
        args.n_calib,
        args.recalibrate,
        output_path=golden_output_path,
    )
    dump_using_mlperf_loadgens(args, sut, dumpfile_path=golden_file_path)

    # Dump mlperf_submission model
    args.model_source = "mlperf_submission"
    mlperf_output_path = "./test/output_rngd_gelu/mlperf"
    sut = get_pytorch_sut(args)
    sut.model = get_quant_model(
        sut,
        args.model_source,
        args.model_script_path,
        args.n_calib,
        args.recalibrate,
        output_path=mlperf_output_path,
    )
    dump_using_mlperf_loadgens(args, sut, dumpfile_path=comparison_file_path)

    # Compare Qparam and logits
    cmp_qparam(args.model_script_path, golden_output_path, mlperf_output_path)

    model_compressor.check_conformance(
        comparison_model=comparison_file_path,
        golden_file_path=golden_file_path,
        mcm_name_to_check="qa_outputs",
        dumping_range="qlv4_linear",
        result_file_path="./compare_pkl",
    )

    print("Done")


if __name__ == "__main__":
    main()
