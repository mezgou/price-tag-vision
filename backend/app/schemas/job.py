from __future__ import annotations

from datetime import datetime

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
    created_at: datetime
    updated_at: datetime
