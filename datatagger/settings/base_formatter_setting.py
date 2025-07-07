from pydantic import Field
from pydantic_settings import BaseSettings


class BaseFormatterSettings(
    BaseSettings, cli_parse_args=True, cli_enforce_required=True
):
    input_file: str = Field(
        ..., description="Input data file path (supports .json/.jsonl)", required=True
    )
    output_file: str = Field(
        ..., description="Output data file path (supports .json/.jsonl)", required=True
    )
    batch_size: int = Field(default=1000, description="Batch size")
    log_level: str = Field(default="INFO", description="Log level")
    prompt_field: str = Field(
        default="instruction", description="Field name in input file to use as prompt"
    )
    output_field: str = Field(
        default="output", description="Field name in input file to use as output"
    )
