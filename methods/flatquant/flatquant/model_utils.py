import torch
import transformers
import logging

from .utils import skip
from .model_tools.llama31_utils import apply_flatquant_to_llama_31


def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def get_llama_31(model_name, seqlen, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = seqlen
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    return model, apply_flatquant_to_llama_31


def get_qwen2(model_name, seqlen, hf_token):
    skip_initialization()
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError:
        logging.error("Qwen2 model is not available in this version of 'transformers'. Please update the library.")
        raise ImportError("Qwen2 model is not available. Ensure you're using a compatible version of the 'transformers' library.")

    config = transformers.Qwen2Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen2ForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = seqlen
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    from .model_tools.qwen_utils import apply_flatquant_to_qwen
    return model, apply_flatquant_to_qwen


# Unified model loading function
def get_model(model_name, seqlen, hf_token=None):
    # 修改使其接受llama其他模型
    # if 'llama-3.1' in model_name.lower() or "DeepSeek-R1-Distill-Llama" in model_name:
    if 'llama' in model_name.lower() or "DeepSeek-R1-Distill-Llama" in model_name:
        return get_llama_31(model_name, seqlen, hf_token)
    elif 'qwen-2.5' in model_name or "DeepSeek-R1-Distill-Qwen" in model_name or "QwQ" in model_name:
        return get_qwen2(model_name, seqlen, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')

