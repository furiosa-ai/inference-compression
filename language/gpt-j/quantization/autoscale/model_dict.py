import transformers
import furiosa_llm_models
import backend_RNGD


#To DO: Add dictionaries for other models in furiosa-llm-models

GPTJForCausalLM_dict = {
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM : transformers.models.gptj.modeling_gptj,
    furiosa_llm_models.gptj.symbolic.huggingface.GPTJForCausalLM : furiosa_llm_models.gptj.symbolic.huggingface,
    furiosa_llm_models.gptj.symbolic.huggingface_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.huggingface_rope,
    furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.huggingface_rope_rngd_gelu,
    furiosa_llm_models.gptj.symbolic.mlperf_submission.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.mlperf_submission,
    backend_RNGD.GPTJForCausalLM: backend_RNGD,
    # furiosa_llm_models.gptj.symbolic.preallocated_concat_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.preallocated_concat_rope,
    furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.paged_attention_rope,
#     furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope,
#      furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope_erf_gelu.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope_erf_gelu,
}