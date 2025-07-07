from datatagger.settings.base_tagger_setting import BaseTaggerSettings
from pydantic import Field


class TaggerSettingsVLLM(BaseTaggerSettings):
    vllm_model_path: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Path to the model when using vllm",
        required=True,
    )
    device: str = Field(default="0", description="CUDA device to use when using vllm")
    tensor_parallel_size: int = Field(
        default=1, description="Tensor parallel size when using vllm"
    )
    dtype: str = Field(default="auto", description="Model data type when using vllm")
    quantization: str = Field(
        default="None", description="Quantization method when using vllm"
    )
    kv_cache_dtype: str = Field(
        default="auto", description="KV cache data type when using vllm"
    )
    max_model_len: int = Field(
        default=4096, description="Maximum model length when using vllm"
    )
    gpu_memory_utilization: float = Field(
        default=0.95, description="GPU memory utilization when using vllm"
    )
