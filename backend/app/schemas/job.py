from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: str
    progress: int
    stage: str | None
    message: str | None
    error: str | None
    input_video_key: str
    output_csv_key: str | None
    preview_json_key: str | None
    crop_keys_json: list[str] | None
    stats_json: dict[str, Any] | None
    pipeline_name: str
    pipeline_version: str
    created_at: datetime
    updated_at: datetime


class JobCropRead(BaseModel):
    key: str
    url: str
    filename: str
