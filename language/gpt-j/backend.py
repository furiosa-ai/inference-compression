import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg
from dataset import Dataset
import yaml
from transformers import AutoConfig
import torch
from torch.utils.data import DataLoader
import json
import quantization
from quantization import calibration_utils
import model_compressor
from dataset import Dataset
import joblib
from generator_RNGD import (MLPerfSubmissionBeamSearch,
                            expand_inputs_for_generation)
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.utils import BeamSearchScorer
from transformers.utils.fx import get_concrete_args
import pickle



gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
}

EARYLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 50256
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 30
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NUM_RETURN_SEQUENCES = 1
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.int8
BUCKET_SIZE = 2048
NUM_REAL_BATCH = 1


class SUT_base():
    def __init__(self, model_path, dtype, dataset_path, max_examples, model_source, num_layers, use_gpu=False, num_splits=1, split_idx=0):
        # TODO : Pass model file name to init instead of args
        print("Loading PyTorch model...")
        self.model_name = "EleutherAI/gpt-j-6B"
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.use_gpu = use_gpu

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            print("BF16 autocast")
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32



            
        if model_source == 'transformers':
            model_cls = AutoModelForCausalLM
            self.gen_source  = 'GenerationMixin'
        else:
            if model_source == 'furiosa_llm_original':
                from furiosa_llm_models.gptj.symbolic.huggingface import GPTJForCausalLM 
                self.gen_source  = 'GenerationMixin'
            elif model_source == 'furiosa_llm_rope':
                from furiosa_llm_models.gptj.symbolic.huggingface_rope import GPTJForCausalLM
                self.gen_source = 'GenerationMixin'
            elif model_source == 'furiosa_llm_rope_rngd_gelu':
                from furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu import GPTJForCausalLM
                self.gen_source = 'GenerationMixin'
            elif model_source == 'paged_attention_rope':
                from furiosa_llm_models.gptj.symbolic.paged_attention_rope import GPTJForCausalLM
                self.gen_source  = 'QuantPagedAttentionGenerator'
            # elif model_source == 'preallocated_concat_rope':
            #     from furiosa_llm_models.gptj.symbolic.preallocated_concat_rope import GPTJForCausalLM
            #     self.gen_source = 'QuantPreAllocatedGenerator'
            # elif model_source == 'paged_attention_optimized_packed':
            #     from furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope import GPTJForCausalLM
            #     self.gen_source = 'PagedAttentionPackedGenerator'
            # elif model_source == 'paged_attention_optimized_packed_erf':
            #     from furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope_erf_gelu import GPTJForCausalLM
            #     self.gen_source = 'PagedAttentionPackedGenerator'
            elif model_source == 'mlperf_submission':
                from furiosa_llm_models.gptj.symbolic.mlperf_submission import GPTJForCausalLM
                self.gen_source = 'MLPerf_submision_generator'

            model_cls = GPTJForCausalLM


        if num_layers > 0:
            from transformers import AutoConfig
            config_exp =  AutoConfig.from_pretrained(self.model_path)
            config_exp.n_layer = num_layers
            self.model = model_cls.from_pretrained(self.model_path, config=config_exp)
        else:
            self.model = model_cls.from_pretrained(
                self.model_path,
                device_map="auto" if not self.use_gpu else None,
                low_cpu_mem_usage=True if not self.use_gpu else False,
                torch_dtype=self.amp_dtype
            )



        # Cast the model to GPU if the flag is set.
        if self.use_gpu:
            print(f"Casting models to GPU...")
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
            self.device = torch.device("cuda:0")
            self.model.to(self.device)

        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1919,
            padding_side="left",
            use_fast=False,)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.data_object = Dataset(
            self.dataset_path, total_count_override=max_examples, num_splits=num_splits, split_idx=split_idx)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        total_samples_done = 0
        list_prompts_tokens = []
        list_prompts_attn_masks = []

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

            # Cast to GPU
            if self.use_gpu:
                input_ids_tensor = input_ids_tensor.to(self.device)
                input_masks_tensor = input_masks_tensor.to(self.device)

            pred_output_batch = self.inference_call(
                input_ids_tensor, input_masks_tensor).cpu().numpy()
            
            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                query_samples[i].id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)

    def inference_call(self, input_ids_tensor, input_masks_tensor):
        ''' Common for all scenarios '''
        torch_device_type = 'cuda' if self.use_gpu else 'cpu'
        with torch.inference_mode(), torch.autocast(device_type=torch_device_type, enabled=self.amp_enabled, dtype=self.amp_dtype if self.amp_enabled else None):
            input_batch = dict()
            input_batch['input_ids'] = input_ids_tensor
            input_batch['attention_mask'] = input_masks_tensor

            # To Do: Implement beam seach for paged attention & preallocated generator
            if self.gen_source == 'GenerationMixin':
                output_batch = self.model.generate(**input_batch, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)
            elif self.gen_source == 'QuantPagedAttentionGenerator':
                output_batch = self.model.generate(input_batch, **gen_kwargs)
            elif self.gen_source == 'QuantPreAllocatedGenerator':  
                output_batch = self.model.generate(input_batch, **gen_kwargs)
            elif self.gen_source == 'MLPerf_submision_generator':
                beam_scorer = BeamSearchScorer(
                    batch_size=input_batch['input_ids'].shape[0],
                    num_beams=NUM_BEAMS,
                    device=input_batch['input_ids'].device,
                    length_penalty=LENGTH_PENALTY,
                    do_early_stopping=EARYLY_STOPPING,
                    num_beam_hyps_to_keep=NUM_RETURN_SEQUENCES,
                    max_length=MAX_LENGTH,
                )
                input_ids_tensor, input_masks_tensor_dict = expand_inputs_for_generation(
                    input_ids=input_batch['input_ids'],
                    expand_size=NUM_BEAMS,
                    attention_mask= input_batch['attention_mask'],
                )
                input_masks_tensor = input_masks_tensor_dict["attention_mask"]

                output_batch = submission_generator.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    beam_scorer=beam_scorer,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    max_length=MAX_LENGTH,
                    pad_token_id=PAD_TOKEN_ID,
                    eos_token_id=EOS_TOKEN_ID,
                    return_dict_in_generate=RETURN_DICT_IN_GENERATE,
                    kv_dtype=KV_DTYPE,
                    bucket_size=BUCKET_SIZE,
                )
                output_batch = self.model.generate(**input_batch, **gen_kwargs)
            else:
                raise NotImplementedError


            input_batch_lengths = [x.shape[0]
                                   for x in input_batch["input_ids"]]

            output_batch_lengths = [x.shape[0] for x in output_batch]

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)


        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, num_layers, use_gpu, num_splits, split_idx, model_source):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, num_layers, use_gpu, num_splits, split_idx, model_source)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, model_source):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, model_source)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, model_source):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, model_source)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(model_path, scenario, dtype, dataset_path, max_examples, model_source, num_layers, use_gpu=False, num_splits=1, split_idx=0):
    if scenario == "Offline":
        return SUT_Offline(model_path, dtype, dataset_path, max_examples, model_source, num_layers, use_gpu, num_splits, split_idx)
    elif scenario == "Server":
        return SUT_Server(model_path, dtype, dataset_path, max_examples, model_source, use_gpu)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model_path, dtype, dataset_path, max_examples, model_source, use_gpu)
