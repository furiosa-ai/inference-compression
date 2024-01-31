from transformers.generation.utils import GenerationMixin
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers.generation.utils import *
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteriaList

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]

class QuantPreTrainedModel(PreTrainedModel):
    def __init__(self, quant_model, config, model_type):
        self.model_type = model_type
        super().__init__(config)
        self.quant_model = quant_model
        self.config = config
    
    def can_generate(self):
        return self.model_type.can_generate()
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.model_type.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.model_type.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.model_type.prepare_inputs_for_generation(self, input_ids, **model_kwargs)
    
    def __call__(self, **kwargs):
        return self.quant_model(**kwargs)
    
   