import transformers
import furiosa_llm_models


#To DO: Add dictionaries for other models in furiosa-llm-models

GPTJForCausalLM_dict = {
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM : transformers.models.gptj.modeling_gptj,
    furiosa_llm_models.gptj.symbolic.huggingface.GPTJForCausalLM : furiosa_llm_models.gptj.symbolic.huggingface,
    furiosa_llm_models.gptj.symbolic.huggingface_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.huggingface_rope,
    furiosa_llm_models.gptj.symbolic.preallocated_concat_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.preallocated_concat_rope,
    furiosa_llm_models.gptj.symbolic.paged_attention_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.paged_attention_rope,
    furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope.GPTJForCausalLM: furiosa_llm_models.gptj.symbolic.paged_attention_optimized_packed_rope,
}