import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PodMemory"
    debug: bool = False
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    llm_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"

    model_config = {"env_prefix": "PM_"}


settings = Settings()
