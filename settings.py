"""
Settings module for application configuration.
"""

from pathlib import Path
from typing import ClassVar, Optional

from pydantic_settings import BaseSettings


class BaseAppSettings(BaseSettings):
    """Base settings class with common configuration."""

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class AuthSettings(BaseAppSettings):
    """Authentication and API keys settings."""

    HUGGINGFACE_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    HF_DATASET_REPO_ID: Optional[str] = None


class DatasetSettings(BaseAppSettings):
    """OpenAI API settings."""

    FEW_SHOT_EXAMPLES_DATASET: str = "Salesforce/xlam-function-calling-60k"
    LLM_MODEL: str = "gpt-4o-mini"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    SINGLE_TOOL_EXAMPLES_PER_TOOL: int = 2
    MULTI_TOOL_EXAMPLES: int = 2
    UNKNOWN_INTENT_EXAMPLES: int = 2
    PARAPHRASE_COUNT: int = 2


class Settings(BaseAppSettings):
    """Main settings class that combines all specialized settings."""

    auth: AuthSettings = AuthSettings()
    dataset: DatasetSettings = DatasetSettings()


# Create global settings instance
settings = Settings()
