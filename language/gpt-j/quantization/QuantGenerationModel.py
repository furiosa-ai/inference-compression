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
from types import SimpleNamespace

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]

@dataclass
class LLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    


class QuantPreTrainedModel(PreTrainedModel):
    def __init__(self, quant_model, model_type, input_names, concrete_args):
        self.model_type = model_type
        super().__init__(quant_model.config)
        self.quant_model = quant_model
        self.config = quant_model.config
        self.input_names = input_names
        self.concrete_args = concrete_args 
        
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
    
    def _reorder_cache(self, 
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self.model_type._reorder_cache(past_key_values, beam_idx)
    
    def __call__(self, **kwargs):
        items_to_delete = []
         
        for key, value in kwargs.items():
            if key in self.concrete_args: #check if the concrete args used when tracing and the elements of kwargs are equal 
                if not value == self.concrete_args[key]:
                    raise ValueError(f"The custom tracer set {key} as {self.concrete_args[key]} but kwargs sets {key} as {value}. Please check the argument again")
                items_to_delete.append(key)
        
        updated_kwargs = {key: value for key, value in kwargs.items() if key not in items_to_delete}

        if "past_key_values" not in updated_kwargs.keys() or updated_kwargs["past_key_values"] == None: #add dummy past_key_valeus
            updated_kwargs["past_key_values"] = tuple([None] * self.config.n_layer)
        
        return LLMOutput(self.quant_model(**updated_kwargs))
    
   
