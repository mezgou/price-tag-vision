from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str = "artifacts"
    pipeline_name: str = "mock"
    pipeline_version: str = "0.1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
