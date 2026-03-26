import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PodMemory"
    debug: bool = False
    openrouter_api_key: str = ""
    llm_model: str = "google/gemma-3-4b-it:free"

    model_config = {"env_prefix": "PM_"}


settings = Settings()

if not settings.openrouter_api_key:
    settings.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
