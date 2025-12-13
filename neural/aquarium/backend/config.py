"""Configuration for the backend bridge."""

import os
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Neural DSL Backend Bridge"
    version: str = "0.3.0"

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: list = ["*"]

    log_level: str = "INFO"

    max_job_output_lines: int = 1000

    api_key: Optional[str] = None

    class Config:
        env_prefix = "NEURAL_"
        env_file = ".env"


settings = Settings()
