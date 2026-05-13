from __future__ import annotations

import logging
import time

from app.db import new_session
from app.models import Job

LOGGER = logging.getLogger("price-tag-vision.worker.tasks")


def process_job(job_id: str) -> dict[str, str]:
    LOGGER.info("Starting job %s.", job_id)

    try:
        _update_job(
            job_id,
            status="running",
            progress=10,
            stage="starting",
            message="Worker picked up the job.",
            error=None,
        )
        time.sleep(2)

        # Placeholder work until the real vision pipeline is connected.
        _update_job(
            job_id,
            status="running",
            progress=60,
            stage="processing",
            message="Processing placeholder pipeline step.",
            error=None,
        )
        time.sleep(2)

        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            stage="completed",
            message="Job completed successfully.",
            error=None,
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
