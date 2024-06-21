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
        default="mlperf_submission",
        type=str,
        choices=[
            "huggingface_rngd_gelu",
            "mlperf_submission",
            "experimental_huggingface_unsplit_packed",
        ],
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


def set_dump_file_path(dump_file_folder):
    if not os.path.exists(dump_file_folder):
        os.makedirs(dump_file_folder)

    golden_dump_file_path = os.path.join(dump_file_folder, "golden.pkl")
    comparison_dump_file_path = os.path.join(dump_file_folder, "comparison.pkl")
    return golden_dump_file_path, comparison_dump_file_path


def set_output_path_for_qformat_qparam(
    model_script_path, recalibrate, output_path
):
    if not recalibrate:
        golden_output_path = "./test/output_rngd_gelu_hf_data/hf"
        comparison_output_path = "./test/output_rngd_gelu_hf_data/mlperf"
        if not is_qparam_same(
            model_script_path, golden_output_path, comparison_output_path
        ):
            raise ValueError(
                "Golden qparam files are corrupted. Retry with --recalibrate option."
            )
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        golden_output_path = os.path.join(output_path, "golden")
        comparison_output_path = os.path.join(output_path, "comparison")

    return golden_output_path, comparison_output_path


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
    # ---------------------------------------------------------
    # Enable debug mode
    # ---------------------------------------------------------
    sut.debug_mode = True
    if args.model_source == "mlperf_submission":
        quant_model = sut.model.model
    else:
        quant_model = sut.model

    model_compressor.set_model_to_dump_golden_model(
        dumpfile_path,
        quant_model,
        dumping_range="qlv4_linear",
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
    )

    lg.StartTestWithLogSettings(
        sut.sut, sut.qsl.qsl, settings, log_settings, args.audit_conf
    )

    test_sample = sut.debug_test_sample

    if sut:
        print("Destroying SUT...")
        lg.DestroySUT(sut.sut)

        print("Destroying QSL...")
        lg.DestroyQSL(sut.qsl.qsl)

    return test_sample


def is_qparam_same(
    model_script_path, golden_output_path, comparison_output_path, print_log=False
):
    import numpy as np

    golden_qparam_path = f"{golden_output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"
    comparison_qparam_path = f"{comparison_output_path}/qparam_{model_script_path.split('.')[1].split('/')[-1]}.npy"

    golden_qparam = np.load(golden_qparam_path, allow_pickle=True).item()
    comparison_qparam = np.load(comparison_qparam_path, allow_pickle=True).item()

    failure_count = 0
    for module_name, module_qparam in comparison_qparam.items():
        try:
            golden_data = golden_qparam[module_name]
        except:
            continue

        for qparam_name, qparam in golden_data.items():
            if qparam is None:
                continue

            if not np.array_equal(module_qparam[qparam_name], qparam):
                failure_count = failure_count + 1
                if print_log:
                    print(
                        "Failed ",
                        module_name,
                        qparam_name,
                        (abs(module_qparam[qparam_name] - qparam) / abs(qparam)).max(),
                    )
            else:
                if print_log:
                    print("Passed ", module_name, qparam_name)

    if failure_count != 0:
        raise ValueError("Qparam comparision test failed.")

    return True


def is_logit_same(
    golden_file_path,
    comparison_model_file_path,
    golden_model_test_sample,
    comparison_model_test_sample,
    mcm_name_to_check,
):
    import pickle

    import torch

    comparison_file = open(comparison_model_file_path, "rb")

    read_golden_file = True

    with open(golden_file_path, "rb") as golden_file:
        while read_golden_file:
            try:
                golden_result = pickle.load(golden_file)
                golden_layer_name = next(iter(golden_result))

                if (
                    mcm_name_to_check is not None
                    and not mcm_name_to_check in golden_layer_name
                ):
                    continue

                while True:
                    comparison_result = pickle.load(comparison_file)
                    comparison_layer_name = next(iter(comparison_result))

                    if golden_layer_name in comparison_layer_name:
                        read_golden_file = False
                        break

            except EOFError:
                print(
                    f"It's end of file. Please check file path {golden_file_path} again."
                )
                break

    golden_result = golden_result[golden_layer_name]
    comparison_result = comparison_result[comparison_layer_name]

    try:
        golden_output = golden_result["output_before_rounding"]
        comparison_output = comparison_result["output_before_rounding"]
    except:  # noqa: E722
        golden_output = golden_result["output"]
        comparison_output = comparison_result["output"]

    if golden_output.dtype != comparison_output.dtype:
        raise ValueError("Invalid values to compare.")

    # ---------------------------------------------------------
    # Masking only valid_seq
    # ---------------------------------------------------------
    golden_input_ids = golden_model_test_sample["input_ids"]
    comparison_input_ids = comparison_model_test_sample["input_ids"]

    batch_size = golden_input_ids.shape[0]

    for batch_idx in range(batch_size):
        max_seq_length = golden_input_ids[batch_idx].shape[0]
        golden_extract_nonzero_locations = torch.nonzero(golden_input_ids[batch_idx])

        golden_valid_seq_length = (
            int(golden_extract_nonzero_locations[-1] + 1)
            - golden_extract_nonzero_locations[0]
        )

        if golden_valid_seq_length == 0:
            raise ValueError("Invalid target locations.")

        # mlperf_submission 모델의 input preprocessing은 generator 내부에서 이루어지므로,
        # golden valid seq length를 기준으로 comparison_output의 valid location 추출.

        comparison_extract_nonzero_locations = [
            (max_seq_length - 1) - golden_valid_seq_length + 1,
            max_seq_length - 1,
        ]

        device = golden_output.device
        valid_golden_output = golden_output[
            batch_idx,
            int(golden_extract_nonzero_locations[0]) : int(
                golden_extract_nonzero_locations[-1] + 1
            ),
        ]
        valid_comparison_output = comparison_output[
            batch_idx,
            int(comparison_extract_nonzero_locations[0]) : int(
                comparison_extract_nonzero_locations[-1] + 1
            ),
        ]

        if not torch.equal(valid_golden_output, valid_comparison_output):
            raise ValueError("Logits comparison test failed.")

    return True


def test_model_equivalence():
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

    golden_output_path, comparison_output_path = set_output_path_for_qformat_qparam(
        args.model_script_path, args.recalibrate, output_path="./test/mlperf_submission"
    )
    golden_dump_file_path, comparison_dump_file_path = set_dump_file_path(
        dump_file_folder="./test/mlperf_submission/dumped"
    )

    # ---------------------------------------------------------
    # Dump golden model
    # ---------------------------------------------------------
    args.model_source = "huggingface_rngd_gelu"
    sut = get_pytorch_sut(args)
    sut.model = get_quant_model(
        sut,
        args.model_source,
        args.model_script_path,
        args.n_calib,
        args.recalibrate,
        output_path=golden_output_path,
    )

    golden_model_test_sample = dump_using_mlperf_loadgens(
        args, sut, dumpfile_path=golden_dump_file_path
    )

    # ---------------------------------------------------------
    # Dump mlperf_submission model
    # ---------------------------------------------------------
    args.model_source = "mlperf_submission"
    sut = get_pytorch_sut(args)
    sut.model = get_quant_model(
        sut,
        args.model_source,
        args.model_script_path,
        args.n_calib,
        args.recalibrate,
        use_packed_dataloader=False,
        output_path=comparison_output_path,
    )
    comparison_model_test_sample = dump_using_mlperf_loadgens(
        args, sut, dumpfile_path=comparison_dump_file_path
    )

    # ---------------------------------------------------------
    # Compare Qparam and logits
    # Bert는 qparam migration 작업을 수행하지 않으므로, qparam 비교 작업도 ci test상에서 수행.
    # ---------------------------------------------------------
    if args.recalibrate:
        if is_qparam_same(
            args.model_script_path, golden_output_path, comparison_output_path
        ):
            print("Qparam comparision test passed.")

    if is_logit_same(
        golden_dump_file_path,
        comparison_dump_file_path,
        golden_model_test_sample,
        comparison_model_test_sample,
        mcm_name_to_check="qa_outputs",
    ):
        print("Logits comparison test passed.")


if __name__ == "__main__":
    test_model_equivalence()
