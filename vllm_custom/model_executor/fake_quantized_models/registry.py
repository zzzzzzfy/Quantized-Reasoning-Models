# 由于npu上暂时不能正常使用fast-hadamard-transform，因此先注释掉未使用的类别注册

from vllm import ModelRegistry
# from vllm_custom.model_executor.fake_quantized_models.llama_fake_quantized import LlamaFakeQuantizedForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.llama_flatquant import LlamaFlatQuantForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.llama_kvquant_star import LlamaKVQuantStarForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.llama_quarot_kv import LlamaQuaRotKVForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.llama_quarot import LlamaQuaRotForCausalLM
from vllm_custom.model_executor.fake_quantized_models.qwen2_fake_quantized import Qwen2FakeQuantizedForCausalLM
from vllm_custom.model_executor.fake_quantized_models.qwen2_flatquant import Qwen2FlatQuantForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.qwen2_kvquant_star import Qwen2KVQuantStarForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.qwen2_quarot_kv import Qwen2QuaRotKVForCausalLM
# from vllm_custom.model_executor.fake_quantized_models.qwen2_quarot import Qwen2QuaRotForCausalLM


def register_fake_quantized_models():
    # ModelRegistry.register_model("LlamaFakeQuantizedForCausalLM", LlamaFakeQuantizedForCausalLM)
    # ModelRegistry.register_model("LlamaFlatQuantForCausalLM", LlamaFlatQuantForCausalLM)
    # ModelRegistry.register_model("LlamaKVQuantStarForCausalLM", LlamaKVQuantStarForCausalLM)
    # ModelRegistry.register_model("LlamaQuaRotKVForCausalLM", LlamaQuaRotKVForCausalLM)
    # ModelRegistry.register_model("LlamaQuaRotForCausalLM", LlamaQuaRotForCausalLM)
    ModelRegistry.register_model("Qwen2FakeQuantizedForCausalLM", Qwen2FakeQuantizedForCausalLM)
    ModelRegistry.register_model("Qwen2FlatQuantForCausalLM", Qwen2FlatQuantForCausalLM)
    # ModelRegistry.register_model("Qwen2KVQuantStarForCausalLM", Qwen2KVQuantStarForCausalLM)
    # ModelRegistry.register_model("Qwen2QuaRotKVForCausalLM", Qwen2QuaRotKVForCausalLM)
    # ModelRegistry.register_model("Qwen2QuaRotForCausalLM", Qwen2QuaRotForCausalLM)

