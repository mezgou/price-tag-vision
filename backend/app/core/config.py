from __future__ import annotations

from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    database_url: str
    redis_url: str
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str = "artifacts"
    vision_service_url: str = Field(
        default="http://vision-service:9001",
        validation_alias=AliasChoices("VISION_SERVICE_URL", "ML_SERVICE_URL"),
    )
    backend_cors_origins: str = ""
    queue_name: str = "jobs"
    worker_heartbeat_prefix: str = "worker:heartbeat"
    worker_heartbeat_ttl_seconds: int = 20

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.backend_cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
