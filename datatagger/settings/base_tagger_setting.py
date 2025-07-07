from enum import Enum, auto
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class TagMission(Enum):
    QUALITY = auto()
    DIFFICULTY = auto()
    CLASSIFICATION = auto()
    SAFETY = auto()
    LANGUAGE = auto()
    REWARD = auto()
    EMBEDDING = auto()


class BaseTaggerSettings(BaseSettings, cli_parse_args=True, cli_enforce_required=True):
    tag_mission: TagMission = Field(
        default=TagMission.QUALITY,
        description="Type of tagging mission to perform",
        required=True,
    )
    enable_thinking: bool = Field(default=False, description="Enable thinking")
    input_file: Optional[str] = Field(
        default=None, description="Input file path, data to be tagged", required=True
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Output file path. If not provided, will use {input_file_base}_{tag_mission}.jsonl",
    )
    prompt_field: str = Field(
        default="instruction", description="Field name in input file to use as prompt"
    )
    output_field: str = Field(
        default="response", description="Field name in input file to use as output"
    )
    batch_size: int = Field(default=100, description="Batch size for processing")
    checkpoint_every: int = Field(default=1000, description="Checkpoint frequency")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, description="Sampling temperature")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    dimension: int = Field(default=2560, description="Embedding dimension")

    milvus_store_embeddings: bool = Field(
        default=False, description="是否将embedding存入Milvus"
    )
    milvus_host: str = Field(default="localhost", description="Milvus服务host")
    milvus_port: str = Field(default="19530", description="Milvus服务端口")
    milvus_collection: str = Field(default="embeddings", description="Milvus集合名")

    # faiss相关配置
    faiss_store_embeddings: bool = Field(
        default=False, description="是否将embedding存入本地Faiss"
    )
    faiss_index_file: str = Field(
        default="data/faiss.index", description="Faiss索引文件路径"
    )
    faiss_meta_file: str = Field(
        default="data/faiss_meta.pkl", description="Faiss meta信息文件路径"
    )
