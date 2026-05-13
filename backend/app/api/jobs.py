from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db_session
from app.models import Job
from app.schemas import JobCropRead, JobRead
from app.services.queue import enqueue_job
from app.services.storage import (
    InputStorage,
    StorageObjectNotFoundError,
    get_input_storage,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
settings = get_settings()


@router.post("", response_model=JobRead, status_code=status.HTTP_201_CREATED)
def create_job(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    storage: InputStorage = Depends(get_input_storage),
) -> Job:
    job_id = str(uuid4())
    original_filename = file.filename or "upload.bin"
    input_video_key = storage.upload_input_file(job_id, file)

    job = Job(
        id=job_id,
        original_filename=original_filename,
        status="uploaded",
        progress=0,
        stage="uploaded",
        message="Video uploaded to object storage.",
        error=None,
        input_video_key=input_video_key,
        pipeline_name=settings.pipeline_name,
        pipeline_version=settings.pipeline_version,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        enqueue_job(job_id)
    except Exception as exc:
        job.status = "failed"
        job.stage = "queue"
        job.message = "Failed to enqueue job."
        job.error = str(exc)
        db.commit()
        db.refresh(job)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue job.",
        ) from exc

    job.status = "queued"
    job.stage = "queued"
    job.message = "Job queued for processing."
    db.commit()
    db.refresh(job)
    return job


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: str, db: Session = Depends(get_db_session)) -> Job:
    return _get_job_or_404(db, job_id)


@router.get("/{job_id}/preview")
def get_job_preview(
    job_id: str,
    db: Session = Depends(get_db_session),
    storage: InputStorage = Depends(get_input_storage),
) -> JSONResponse:
    job = _get_job_or_404(db, job_id)
    _ensure_job_succeeded(job)

    if not job.preview_json_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview artifact not found.",
        )

    try:
        payload = storage.get_json(job.preview_json_key)
    except StorageObjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview artifact not found.",
        ) from exc

    return JSONResponse(content=payload)


@router.get("/{job_id}/download/csv")
def download_job_csv(
    job_id: str,
    db: Session = Depends(get_db_session),
    storage: InputStorage = Depends(get_input_storage),
) -> StreamingResponse:
    job = _get_job_or_404(db, job_id)
    _ensure_job_succeeded(job)

    if not job.output_csv_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CSV artifact not found.",
        )

    try:
        payload = storage.get_bytes(job.output_csv_key)
    except StorageObjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CSV artifact not found.",
        ) from exc

    return StreamingResponse(
        iter([payload]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="result_{job.id}.csv"',
        },
    )


@router.get("/{job_id}/crops", response_model=list[JobCropRead])
def get_job_crops(
    job_id: str,
    db: Session = Depends(get_db_session),
    storage: InputStorage = Depends(get_input_storage),
) -> list[JobCropRead]:
    job = _get_job_or_404(db, job_id)
    crop_keys = job.crop_keys_json or []
    return [
        JobCropRead(
            key=key,
            url=storage.generate_presigned_url(key, expires_in=3600),
            filename=Path(key).name,
        )
        for key in crop_keys
    ]


def _get_job_or_404(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    return job


def _ensure_job_succeeded(job: Job) -> None:
    if job.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job result is not ready.",
        )
