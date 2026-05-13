from __future__ import annotations

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.pipelines.mock import build_mock_artifacts
from app.services.storage import ArtifactStorage

settings = get_settings()
storage = ArtifactStorage(settings)

app = FastAPI(
    title="Price Tag Vision Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


class ProcessRequest(BaseModel):
    job_id: str
    input_video_key: str
    pipeline_name: str = Field(default="mock")
    pipeline_version: str = Field(default="0.1.0")
    config: dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    csv_key: str
    preview_key: str
    crop_keys: list[str]
    stats: dict[str, Any]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "vision-service",
        "process_url": "/api/pipeline/process",
        "health_url": "/health",
    }


@app.get("/health")
def health() -> JSONResponse:
    try:
        storage.ensure_bucket()
    except (
        BotoCoreError,
        ClientError,
    ) as exc:  # pragma: no cover - defensive runtime check
        return JSONResponse(
            {
                "service": "vision-service",
                "status": "degraded",
                "detail": f"MinIO bucket check failed: {exc}",
            },
            status_code=503,
        )

    return JSONResponse(
        {
            "service": "vision-service",
            "status": "ok",
            "detail": f"Bucket '{settings.s3_bucket}' is available.",
        }
    )


@app.post("/api/pipeline/process", response_model=ProcessResponse)
def process(request: ProcessRequest) -> ProcessResponse:
    try:
        storage.ensure_object(request.input_video_key)
    except ClientError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Input video key was not found in MinIO: {request.input_video_key}",
        ) from exc

    artifacts = build_mock_artifacts(
        job_id=request.job_id,
        input_video_key=request.input_video_key,
        pipeline_name=request.pipeline_name,
        pipeline_version=request.pipeline_version,
    )

    csv_key = f"outputs/{request.job_id}/result.csv"
    preview_key = f"outputs/{request.job_id}/preview.json"
    crop_key = f"outputs/{request.job_id}/crops/crop_001.jpg"

    storage.upload_bytes(csv_key, artifacts.csv_bytes, "text/csv; charset=utf-8")
    storage.upload_bytes(preview_key, artifacts.preview_bytes, "application/json")
    storage.upload_bytes(crop_key, artifacts.crop_bytes, "image/jpeg")

    return ProcessResponse(
        job_id=request.job_id,
        status="succeeded",
        csv_key=csv_key,
        preview_key=preview_key,
        crop_keys=[crop_key],
        stats=artifacts.stats,
    )
