from transformers import PreTrainedModel
from furiosa_llm_models.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers.utils.fx import HFTracer, get_concrete_args, check_if_model_is_supported
import torch
from torch.fx import Tracer, GraphModule
from typing import Any, Callable, Dict, List, Optional, Type, Union


def get_input_names_and_concrete_args(model: PreTrainedModel, prefill_mode = True):
    module = type(model).__module__.split('.')[0]
    if module not in ['transformers', 'furiosa_llm_models']:
        raise NotImplementedError
    
    #defined according to MLperf models
    custom_concrete_args = {'use_cache': True, 'return_dict': True,
                                    'output_attentions': False, 'output_hidden_states': False}
    if prefill_mode:
        input_names = ["input_ids", "position_ids", "attention_mask"]
    else:
        input_names = ["input_ids", "past_key_values", "position_ids", "attention_mask"]
        



    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    for args in custom_concrete_args.keys():
        if args not in concrete_args.keys():
            raise ValueError(f'{args} does not belong to {concrete_args}.')
        concrete_args[args] = custom_concrete_args[args]

    return input_names, concrete_args


def custom_symbolic_trace(model: PreTrainedModel, prefill_mode = True, disable_check: bool = False) -> GraphModule:
    
    input_names, concrete_args = get_input_names_and_concrete_args(model, prefill_mode)

    if not disable_check:
        check_if_model_is_supported(model)

    # Tracing.
    tracer = HFTracer()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced, input_names, concrete_args
