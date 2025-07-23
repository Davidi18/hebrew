"""
Configuration settings for Hebrew Content Intelligence Service.
Uses Pydantic Settings for environment variable management.
"""

from typing import Optional
from pydantic import Field, BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Service Configuration
    service_name: str = Field(default="hebrew-content-intelligence", alias="SERVICE_NAME")
    version: str = Field(default="1.0.0", alias="VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=5000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    
    # HebSpacy Configuration
    hebspacy_model: str = Field(default="he", alias="HEBSPACY_MODEL")
    hebspacy_cache_dir: str = Field(default="./models/model_cache", alias="HEBSPACY_CACHE_DIR")
    
    # DataForSEO Integration
    dataforseo_api_url: Optional[str] = Field(default=None, alias="DATAFORSEO_API_URL")
    dataforseo_api_key: Optional[str] = Field(default=None, alias="DATAFORSEO_API_KEY")
    
    # Caching Configuration
    cache_type: str = Field(default="memory", alias="CACHE_TYPE")  # memory, redis, disk
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")  # seconds
    
    # Performance Settings
    max_content_length: int = Field(default=50000, alias="MAX_CONTENT_LENGTH")
    request_timeout: int = Field(default=30, alias="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=10, alias="MAX_CONCURRENT_REQUESTS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
