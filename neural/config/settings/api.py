"""
API server configuration settings.

DEPRECATED: The API server module has been removed as per v0.4.0.
This configuration file is retained for backward compatibility but is no longer used.
For REST API functionality, wrap Neural in FastAPI/Flask yourself.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, field_validator

from neural.config.base import BaseConfig


class APISettings(BaseConfig):
    """API server settings."""
    
    model_config = {"env_prefix": "NEURAL_API_"}
    
    # Application settings
    app_name: str = Field(default="Neural DSL API", description="API name")
    app_version: str = Field(default="0.3.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, gt=0, lt=65536, description="Server port")
    workers: int = Field(default=4, gt=0, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Security settings
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT and encryption"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    access_token_expire_minutes: int = Field(
        default=60,
        gt=0,
        description="Access token expiration in minutes"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        gt=0,
        description="Maximum requests per period"
    )
    rate_limit_period: int = Field(
        default=60,
        gt=0,
        description="Rate limit period in seconds"
    )
    
    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(
        default=6379,
        gt=0,
        lt=65536,
        description="Redis port"
    )
    redis_db: int = Field(default=0, ge=0, description="Redis database")
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    
    # Celery settings
    celery_broker_url: Optional[str] = Field(
        default=None,
        description="Celery broker URL"
    )
    celery_result_backend: Optional[str] = Field(
        default=None,
        description="Celery result backend URL"
    )
    celery_task_timeout: int = Field(
        default=3600,
        gt=0,
        description="Celery task timeout in seconds"
    )
    
    # Database settings
    database_url: str = Field(
        default="sqlite:///./neural_api.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries"
    )
    database_pool_size: int = Field(
        default=5,
        gt=0,
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10,
        gt=0,
        description="Maximum database connection overflow"
    )
    
    # Webhook settings
    webhook_timeout: int = Field(
        default=30,
        gt=0,
        description="Webhook timeout in seconds"
    )
    webhook_retry_limit: int = Field(
        default=3,
        ge=0,
        description="Webhook retry limit"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    cors_methods: List[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def broker_url(self) -> str:
        """Get Celery broker URL."""
        return self.celery_broker_url or self.redis_url
    
    @property
    def result_backend(self) -> str:
        """Get Celery result backend URL."""
        return self.celery_result_backend or self.redis_url
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle comma-separated string
            if v.startswith("["):
                # Handle JSON-like string
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in v.split(",")]
        return v
