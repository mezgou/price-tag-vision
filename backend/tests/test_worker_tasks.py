from __future__ import annotations

import logging

import pytest

from app.models import Job
import app.worker.tasks as worker_tasks


def _create_job(session_factory, **overrides) -> Job:
    payload = {
        "id": "job-worker-1",
        "original_filename": "input.mp4",
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "message": "Job queued for processing.",
        "error": None,
        "input_video_key": "inputs/job-worker-1/input.mp4",
    }
    payload.update(overrides)

    with session_factory() as session:
        job = Job(**payload)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def test_process_job_transitions_job_to_succeeded(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _create_job(session_factory)
    updates: list[dict[str, str | int | None]] = []
    original_update_job = worker_tasks._update_job

    monkeypatch.setattr(worker_tasks, "new_session", session_factory)
    monkeypatch.setattr(worker_tasks.time, "sleep", lambda _: None)

    def track_updates(job_id: str, **changes: str | int | None) -> None:
        updates.append(changes.copy())
        original_update_job(job_id, **changes)

    monkeypatch.setattr(worker_tasks, "_update_job", track_updates)

    result = worker_tasks.process_job(job.id)

    assert result == {"job_id": job.id, "status": "succeeded"}
    assert [update["status"] for update in updates] == [
        "running",
        "running",
        "succeeded",
    ]
    assert [update["stage"] for update in updates] == [
        "starting",
        "processing",
        "completed",
    ]
    assert updates[-1]["progress"] == 100

    with session_factory() as session:
        refreshed_job = session.get(Job, job.id)
        assert refreshed_job is not None
        assert refreshed_job.status == "succeeded"
        assert refreshed_job.progress == 100
        assert refreshed_job.stage == "completed"
        assert refreshed_job.message == "Job completed successfully."


def test_process_job_logs_missing_job_failure(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(worker_tasks, "new_session", session_factory)
    monkeypatch.setattr(worker_tasks.time, "sleep", lambda _: None)
    caplog.set_level(logging.ERROR, logger="price-tag-vision.worker.tasks")

    with pytest.raises(RuntimeError, match="missing-job"):
        worker_tasks.process_job("missing-job")

    assert "Job missing-job failed." in caplog.text
    assert "Failed to persist failure state for job missing-job." in caplog.text
