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
    s3_public_endpoint_url: str | None = None
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str = "artifacts"
    vision_service_url: str = Field(
        default="http://vision-service:9001",
        validation_alias=AliasChoices("VISION_SERVICE_URL", "ML_SERVICE_URL"),
    )
    # 24h == effectively no limit. A full far-4K video pass legitimately runs
    # 20-40 min; the old 900s cut the worker->vision HTTP call mid-run and
    # produced truncated CSVs that did not match a local full run.
    vision_service_timeout_seconds: float = Field(
        default=86400.0,
        validation_alias=AliasChoices("VISION_SERVICE_TIMEOUT_SECONDS"),
    )
    pipeline_name: str = Field(
        default="price_tag_cpu_v1",
        validation_alias=AliasChoices("PIPELINE_NAME"),
    )
    pipeline_version: str = Field(
        default="0.1.0",
        validation_alias=AliasChoices("PIPELINE_VERSION"),
    )
    backend_cors_origins: str = ""
    queue_name: str = "jobs"
    # -1 == RQ "infinite" job timeout. Video processing time scales with
    # footage length; a hard wall just kills otherwise-valid long jobs.
    job_timeout_seconds: int = Field(
        default=-1,
        validation_alias=AliasChoices("JOB_TIMEOUT_SECONDS"),
    )
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
