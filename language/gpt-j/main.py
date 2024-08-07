import subprocess
import mlperf_loadgen as lg
import argparse
import os
import sys
from backend import get_SUT
import quantization 

import time
from datetime import timedelta
from utils import set_optimization, random_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument(
        "--dataset-path", default="./data/cnn_eval.json", help="")
    parser.add_argument(
        "--calib-dataset-path", default="./data/cnn_dailymail_calibration.json", help="")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--dtype", default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true",
                        help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU instead of CPU for the inference")
    parser.add_argument("--audit_conf", default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--model_script_path", default="./quantization/model_script/Qlevel1_RGDA0-W8A16-PTQ.yaml", help="")
    parser.add_argument("--use_mcp", action="store_true", default=False, help="use mcp to quantize the model")
    parser.add_argument("--recalibrate", action="store_true", default=False, help="load already existing quantization metadata")
    parser.add_argument("--num_splits", type=int, default=1, help="")
    parser.add_argument("--split_idx", type=int, default=0, help="")
    parser.add_argument('--torch_optim',default='none',type=str,choices=['default', 'none'],help='Torch optimization.',)
    parser.add_argument("--model_source", type = str, default = "furiosa_llm_original", help="the type of GPTJForCausalLM to use")
    parser.add_argument('--n_layers',default='-1',type=int, help='set the number of layers.',)
    parser.add_argument('--calib_without_padding', action="store_true", default=False, help="use mcp to quantize the model")
    parser.add_argument('--qformat_path', type = str, default=None, help="")
    parser.add_argument('--qparam_path', type = str, default=None, help="")
    
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main():
    args = get_args()
    set_optimization(args)
    random_seed()
    sut = get_SUT(
        model_path=args.model_path,
        scenario=args.scenario,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
        use_gpu=args.gpu,
        num_splits=args.num_splits,
        split_idx=args.split_idx,
        model_source = args.model_source,
        num_layers = args.n_layers, 
    )

    if args.use_mcp:
        sut.model = quantization.get_quant_model(sut.model, args.calib_dataset_path, args.model_script_path, args.calib_without_padding, args.recalibrate, args.qformat_path, args.qparam_path)

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    log_path = os.environ.get("LOG_PATH")
    if args.model_script_path != "":
        if "fp32" in args.model_script_path or args.use_mcp == False:
            if args.num_splits > 1:
                log_path = f"build/logs/fp32/{args.dataset_path.split('.')[1].split('/')[-1]}_{args.num_splits}_{args.split_idx}"
            else:
                log_path = f"build/logs/fp32/{args.dataset_path.split('.')[1].split('/')[-1]}"
        else:
            if args.num_splits > 1:
                log_path = f"build/logs/{args.model_script_path.split('.')[1].split('/')[-1]}/{args.dataset_path.split('.')[1].split('/')[-1]}_{args.num_splits}_{args.split_idx}"
            else:
                log_path = f"build/logs/{args.model_script_path.split('.')[1].split('/')[-1]}/{args.dataset_path.split('.')[1].split('/')[-1]}"
    else:
        log_path = f"build/logs/{args.dataset_path.split('.')[1].split('/')[-1]}"
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
    start_time = time.time()
    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings, args.audit_conf)

    end_time = time.time()
    time_seconds = end_time - start_time
    time_hour = timedelta(seconds=time_seconds)
    print(f"Test running time: {time_hour}")
    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
