import subprocess
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
from SUT import SUT, SUTServer
from quantization.utils import random_seed, set_optimization 

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-MAIN")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="Model name")
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
    parser.add_argument("--dtype", type=str, default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--device", type=str,  choices=["cpu", "cuda:0"], default="cuda:0", help="device to use")
    parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--mlperf-conf", type=str, default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--total-sample-count", type=int, default=24576, help="Number of samples to use in benchmark.") # TODO: This interpretation of 'total-sample-count' is a little misleading. Fix it
    parser.add_argument("--output-log-dir", type=str, default="output-logs", help="Where logs are saved")
    parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to process queries")
    parser.add_argument("--model_source", 
                        choices=[
                            "furiosa_llm_rope", 
                            "preallocated_concat_rope", 
                            "mlperf_submission"
                            ], 
                        default="furiosa_llm_rope", 
                        help="the type of GPTJForCausalLM to use")
    parser.add_argument("--quantize", action="store_true", help="quantize model using ModelComPressor(MCP)")
    parser.add_argument("--quant_config_path", help="a config for model quantization")
    parser.add_argument("--quant_param_path", help="quantization parameters for calibraed layers")
    parser.add_argument("--quant_format_path", help="quantization specifications for calibrated layers")
    parser.add_argument('--torch_numeric_optim', action="store_true", help="use Pytorch numerical optimizaiton for CUDA/cudnn")
    parser.add_argument("--n_layers", type=int, default=-1, help="Set the number of layers of the model.")
    parser.add_argument("--weighted_op_emul_dtype", type=str, default="fp32", help="set emulation type of weighted operators")
    parser.add_argument("--num_splits", type=int, default=1, help="")
    parser.add_argument("--split_idx", type=int, default=0, help="")

    args = parser.parse_args()
    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    }

sut_map = {
        "offline": SUT,
        "server": SUTServer
        }

def main():
    args = get_args()
    
    random_seed()
    set_optimization(args.torch_numeric_optim)
    
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "llama2-70b", args.scenario)
    settings.FromConfig(args.user_conf, "llama2-70b", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    sut_cls = sut_map[args.scenario.lower()]

    sut = sut_cls(
        model_path=args.model_path,
        model_source=args.model_source,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        device=args.device,
        n_layers=args.n_layers,
        num_splits=args.num_splits,
        split_idx=args.split_idx,
    )
    
    if args.quantize:
        import quantization 
        sut.model = quantization.get_quant_model(sut.model, args)
    
    
    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings, args.audit_conf)

    # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
