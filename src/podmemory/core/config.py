import os

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PodMemory"
    debug: bool = False
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    llm_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"

    model_config = {"env_prefix": "PM_"}

    @field_validator("groq_api_key")
    @classmethod
    def groq_key_required(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Set GROQ_API_KEY or PM_GROQ_API_KEY environment variable."
            )
        return v


settings = Settings()
