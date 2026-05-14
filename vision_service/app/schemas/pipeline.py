from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    job_id: str
    input_video_key: str
    pipeline_name: str | None = Field(default=None)
    pipeline_version: str | None = Field(default=None)
    config: dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    csv_key: str
    preview_key: str
    crop_keys: list[str]
    stats: dict[str, Any]
