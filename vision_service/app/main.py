from __future__ import annotations

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.pipelines.base import PipelineExecutionError
from app.pipelines.registry import get_pipeline_registry
from app.schemas.pipeline import ProcessRequest, ProcessResponse
from app.services.storage import ArtifactStorage

settings = get_settings()
storage = ArtifactStorage(settings)
registry = get_pipeline_registry()

app = FastAPI(
    title="Price Tag Vision Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


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

    requested_pipeline_name = request.pipeline_name or settings.pipeline_name
    pipeline = registry.resolve(
        requested_pipeline_name,
        default_name=settings.pipeline_name,
    )
    resolved_request = request.model_copy(
        update={
            "pipeline_name": pipeline.name,
            "pipeline_version": request.pipeline_version or pipeline.default_version,
        }
    )

    try:
        return pipeline.run(resolved_request, storage)
    except PipelineExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
