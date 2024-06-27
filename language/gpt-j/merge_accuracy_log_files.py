from dataset import Dataset
import os
import time
import numpy as np
import json
import nltk
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import evaluate
import argparse
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml,pdb

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_script_name", required=True,
                        help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file", default="./data/cnn_eval.json",
                        help="path to cnn_eval.json")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose messages")
    parser.add_argument("--dtype", default="int64",
                        help="dtype of the accuracy log", choices=["int32", "int64"])
    parser.add_argument("--num_splits", type=int, default=1, 
                        help="")
    args = parser.parse_args()
    return args


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def main():

    args = get_args()
    model_name = "EleutherAI/gpt-j-6B"
    dataset_path = args.dataset_file
    metric = evaluate.load("rouge")
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,)
    tokenizer.pad_token = tokenizer.eos_token

    root_logdir = os.path.join('./build/logs', args.model_script_name)
    default_log_name="mlperf_log_accuracy.json"
    results_path_list = []
    for idx in range(args.num_splits):
        results_path_list.append(os.path.join(root_logdir, f"{args.dataset_file.split('.')[1].split('/')[-1]}_{args.num_splits}_{idx}"))
        
    

    results = []

    for result_path in results_path_list:
        with open(os.path.join(result_path, default_log_name), "r") as f:
            result = json.load(f)
            results.append(result)
        

    n_splited_data=len(results[0])
    # Deduplicate the results loaded from the json
    dedup_results = []
    seen = set()
    for idx, split_result in enumerate(results):
        for result in split_result:
            item = result['qsl_idx'] + n_splited_data*idx
            seq_id = result['seq_id']+ n_splited_data*idx
            if item not in seen:
                seen.add(item)
                result['qsl_idx'] = item
                result['seq_id'] = seq_id
                dedup_results.append(result)



    results = dedup_results      
    with open('mlperf_log_accuracy.json', 'w') as file:
        json.dump(results, file)
    
    


if __name__ == "__main__":
    main()
