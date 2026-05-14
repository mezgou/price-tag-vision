from __future__ import annotations

import logging

import pytest

from app.models import Job
from app.services.vision_client import VisionProcessResponse, VisionServiceError
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
    monkeypatch.setattr(
        worker_tasks,
        "process_video_job",
        lambda **_: VisionProcessResponse(
            job_id=job.id,
            status="succeeded",
            csv_key=f"outputs/{job.id}/result.csv",
            preview_key=f"outputs/{job.id}/preview.json",
            crop_keys=[f"outputs/{job.id}/crops/crop_001.jpg"],
            stats={
                "detected_price_tags": 1,
                "pipeline_name": "price_tag_cpu_v1",
                "pipeline_version": "0.1.0",
            },
        ),
    )

    def track_updates(job_id: str, **changes: str | int | None) -> None:
        updates.append(changes.copy())
        original_update_job(job_id, **changes)

    monkeypatch.setattr(worker_tasks, "_update_job", track_updates)

    result = worker_tasks.process_job(job.id)

    assert result == {"job_id": job.id, "status": "succeeded"}
    assert [update["status"] for update in updates] == [
        "running",
        "succeeded",
    ]
    assert [update["stage"] for update in updates] == [
        "calling_vision_service",
        "completed",
    ]
    assert updates[-1]["progress"] == 100
    assert updates[-1]["output_csv_key"] == f"outputs/{job.id}/result.csv"
    assert updates[-1]["preview_json_key"] == f"outputs/{job.id}/preview.json"
    assert updates[-1]["crop_keys_json"] == [f"outputs/{job.id}/crops/crop_001.jpg"]
    assert updates[-1]["stats_json"] == {
        "detected_price_tags": 1,
        "pipeline_name": "price_tag_cpu_v1",
        "pipeline_version": "0.1.0",
    }

    with session_factory() as session:
        refreshed_job = session.get(Job, job.id)
        assert refreshed_job is not None
        assert refreshed_job.status == "succeeded"
        assert refreshed_job.progress == 100
        assert refreshed_job.stage == "completed"
        assert refreshed_job.message == "Vision pipeline completed"
        assert refreshed_job.output_csv_key == f"outputs/{job.id}/result.csv"
        assert refreshed_job.preview_json_key == f"outputs/{job.id}/preview.json"
        assert refreshed_job.crop_keys_json == [f"outputs/{job.id}/crops/crop_001.jpg"]
        assert refreshed_job.stats_json == {
            "detected_price_tags": 1,
            "pipeline_name": "price_tag_cpu_v1",
            "pipeline_version": "0.1.0",
        }


def test_process_job_logs_missing_job_failure(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(worker_tasks, "new_session", session_factory)
    caplog.set_level(logging.ERROR, logger="price-tag-vision.worker.tasks")

    with pytest.raises(RuntimeError, match="missing-job"):
        worker_tasks.process_job("missing-job")

    assert "Job missing-job failed." in caplog.text
    assert "Failed to persist failure state for job missing-job." in caplog.text


def test_process_job_marks_job_failed_when_vision_service_errors(
    session_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _create_job(session_factory, id="job-worker-failed")

    monkeypatch.setattr(worker_tasks, "new_session", session_factory)
    monkeypatch.setattr(
        worker_tasks,
        "process_video_job",
        lambda **_: (_ for _ in ()).throw(
            VisionServiceError("Vision service unavailable.")
        ),
    )

    with pytest.raises(VisionServiceError, match="Vision service unavailable."):
        worker_tasks.process_job(job.id)

    with session_factory() as session:
        refreshed_job = session.get(Job, job.id)
        assert refreshed_job is not None
        assert refreshed_job.status == "failed"
        assert refreshed_job.stage == "failed"
        assert refreshed_job.message == "Job failed during worker execution."
        assert refreshed_job.error == "Vision service unavailable."
        assert refreshed_job.progress == 10
