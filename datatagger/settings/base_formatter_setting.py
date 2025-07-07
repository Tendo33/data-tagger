from pydantic import Field
from pydantic_settings import BaseSettings


class BaseFormatterSettings(
    BaseSettings, cli_parse_args=True, cli_enforce_required=True
):
    input_file: str = Field(
        ..., description="输入数据文件路径（支持 .json/.jsonl）", required=True
    )
    output_file: str = Field(
        ..., description="输出数据文件路径（支持 .json/.jsonl）", required=True
    )

    log_level: str = Field(default="INFO", description="日志等级")
    prompt_field: str = Field(
        default="instruction", description="Field name in input file to use as prompt"
    )
    output_field: str = Field(
        default="response", description="Field name in input file to use as output"
    )
