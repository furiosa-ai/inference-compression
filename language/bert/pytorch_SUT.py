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

import array
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"
    ),
)
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
from transformers import BertConfig

from squad_QSL import get_squad_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
            config_json = json.load(f)

        if args.n_layers > 0:
            config_json["num_hidden_layers"] = args.n_layers

        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"],
        )

        self.network = args.network
        self.dev = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.version = transformers.__version__
        self.model_source = args.model_source
        self.debug_mode = False
        self.debug_test_sample = None

        print("Loading PyTorch model...")

        if self.model_source == "huggingface_rngd_gelu":
            from furiosa_llm_models.bert.symbolic.huggingface_rngd_gelu import (
                BertForQuestionAnswering,
            )
        elif self.model_source == "mlperf_submission":
            from furiosa_llm_models.bert.symbolic.mlperf_submission import (
                BertForQuestionAnswering,
            )
        elif self.model_source == "experimental_huggingface_unsplit_packed":
            from furiosa_llm_models.bert.symbolic.experimental.huggingface_unsplit_packed import (BertForQuestionAnswering)

        self.model = BertForQuestionAnswering(config)
        self.model.to(self.dev)
        self.model.eval()
        model_file = os.environ.get(
            "ML_MODEL_FILE_WITH_PATH",
            "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
        )
        self.model.load_state_dict(torch.load(model_file), strict=False)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            self.process_sample(eval_features, query_samples[i].id)

    def save_test_sample(self, sample: dict):
        if self.debug_test_sample is None:
            self.debug_test_sample = sample

    def process_sample(self, sample_input, query_id=None):
        if self.network == "sut":
            input_ids = sample_input["input_ids"]
            input_mask = sample_input["input_mask"]
            segment_ids = sample_input["segment_ids"]
        else:
            input_ids = sample_input.input_ids
            input_mask = sample_input.input_mask
            segment_ids = sample_input.segment_ids

        # Reformatting
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.dev)
        attention_mask = torch.LongTensor(input_mask).unsqueeze(0).to(self.dev)
        token_type_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(self.dev)

        with torch.no_grad():
            if self.model_source == "huggingface_rngd_gelu":
                if self.debug_mode:
                    # 첫번째 샘플에 대해서 비교 진행
                    self.save_test_sample(
                        sample={
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                        }
                    )

                model_output = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            elif self.model_source in ['mlperf_submission',  'experimental_huggingface_unsplit_packed']:
                padded_sequences = {}
                padded_sequences["input_ids"] = input_ids
                padded_sequences["attention_mask"] = attention_mask
                padded_sequences["token_type_ids"] = token_type_ids



                if self.debug_mode:
                    # 첫번째 샘플에 대해서 비교 진행
                    self.save_test_sample(
                        sample={
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                        }
                    )

                model_output = self.model.generate(
                    **padded_sequences,
                    bucket_size=384,
                    pad_token_id=0,
                )

            if self.version >= "4.0.0":
                if self.model_source == "huggingface_rngd_gelu":
                    start_scores = model_output["start_logits"]
                    end_scores = model_output["end_logits"]
                elif self.model_source in ['mlperf_submission',  'experimental_huggingface_unsplit_packed']:
                    start_scores = model_output[0][:, 0]  # batch_size=1 가정.
                    end_scores = model_output[0][:, 1]
                else:
                    NotImplemented

            else:
                start_scores, end_scores = model_output
            output = (
                torch.stack([start_scores, end_scores], axis=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            if self.network == "sut":
                return output.tolist()

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
