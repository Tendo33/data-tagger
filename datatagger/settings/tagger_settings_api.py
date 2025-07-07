from datatagger.settings.base_tagger_setting import BaseTaggerSettings
from pydantic import Field


class TaggerSettingsAPI(BaseTaggerSettings):
    api_model_name: str = Field(
        default="Meta-Llama-3-8B-Instruct", description="Model name", required=True
    )
    api_url: str = Field(default="", description="API URL for remote model")
    api_key: str = Field(default="", description="API key for remote model")
