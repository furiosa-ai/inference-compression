import transformers
import furiosa_llm_models


#To DO: Add dictionaries for other models in furiosa-llm-models

GPTJForCausalLM_dict = {
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM : transformers.models.gptj.modeling_gptj,
    furiosa_llm_models.gptj.huggingface.GPTJForCausalLM : furiosa_llm_models.gptj.huggingface,
    furiosa_llm_models.gptj.paged_attention_concat.GPTJForCausalLM : furiosa_llm_models.gptj.paged_attention_concat,
    furiosa_llm_models.gptj.huggingface_rope.GPTJForCausalLM: furiosa_llm_models.gptj.huggingface_rope,
    furiosa_llm_models.gptj.paged_attention_concat_rope.GPTJForCausalLM: furiosa_llm_models.gptj.paged_attention_concat_rope,
    furiosa_llm_models.gptj.preallocated_concat_rope.GPTJForCausalLM: furiosa_llm_models.gptj.preallocated_concat_rope,
    furiosa_llm_models.gptj.paged_attention_rope.GPTJForCausalLM: furiosa_llm_models.gptj.paged_attention_rope,
}