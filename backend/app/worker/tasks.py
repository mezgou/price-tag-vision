from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import get_settings
from app.db import new_session
from app.models import Job
from app.services.vision_client import process_video_job

LOGGER = logging.getLogger("price-tag-vision.worker.tasks")


@dataclass(slots=True)
class JobContext:
    id: str
    input_video_key: str
    pipeline_name: str | None
    pipeline_version: str | None


def process_job(job_id: str) -> dict[str, str]:
    LOGGER.info("Starting job %s.", job_id)

    try:
        settings = get_settings()
        job = _get_job_context(job_id)
        pipeline_name = job.pipeline_name or settings.pipeline_name
        pipeline_version = job.pipeline_version or settings.pipeline_version

        _update_job(
            job_id,
            status="running",
            progress=10,
            stage="calling_vision_service",
            message="Calling vision pipeline.",
            error=None,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )

        response = process_video_job(
            job_id=job.id,
            input_video_key=job.input_video_key,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            config={},
        )

        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            stage="completed",
            message="Vision pipeline completed",
            error=None,
            output_csv_key=response.csv_key,
            preview_json_key=response.preview_key,
            crop_keys_json=response.crop_keys,
            stats_json=response.stats,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
        )
        LOGGER.info("Completed job %s.", job_id)
        return {"job_id": job_id, "status": "succeeded"}
    except Exception as exc:
        LOGGER.exception("Job %s failed.", job_id)
        _mark_job_failed(job_id, str(exc))
        raise


def _update_job(job_id: str, **changes: str | int | None) -> None:
    with new_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError(f"Job '{job_id}' was not found.")

        for field_name, value in changes.items():
            setattr(job, field_name, value)

        session.commit()


def _mark_job_failed(job_id: str, error: str) -> None:
    try:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Job failed during worker execution.",
            error=error,
        )
    except Exception:
        LOGGER.exception("Failed to persist failure state for job %s.", job_id)


def _get_job_context(job_id: str) -> JobContext:
    with new_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError(f"Job '{job_id}' was not found.")

        return JobContext(
            id=job.id,
            input_video_key=job.input_video_key,
            pipeline_name=job.pipeline_name,
            pipeline_version=job.pipeline_version,
        )
