from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.models import Job
from app.schemas import JobRead
from app.services.queue import enqueue_job
from app.services.storage import InputStorage, get_input_storage

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
    return job
