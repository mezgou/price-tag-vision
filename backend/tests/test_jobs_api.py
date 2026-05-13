from __future__ import annotations

from uuid import UUID

import app.api.jobs as jobs_api
from app.main import app
from app.models import Job
from app.schemas import JobRead
from app.services.storage import get_input_storage


def _create_job(session_factory, **overrides) -> Job:
    payload = {
        "id": "job-123",
        "original_filename": "input.mp4",
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "message": "Job queued for processing.",
        "error": None,
        "input_video_key": "inputs/job-123/input.mp4",
    }
    payload.update(overrides)

    with session_factory() as session:
        job = Job(**payload)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def test_job_model_persists_defaults_and_serializes(session_factory) -> None:
    with session_factory() as session:
        job = Job(
            id="job-model-1",
            original_filename="clip.mov",
            status="queued",
            stage="queued",
            message="Job queued for processing.",
            error=None,
            input_video_key="inputs/job-model-1/input.mov",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        payload = JobRead.model_validate(job)

    assert job.progress == 0
    assert job.created_at is not None
    assert job.updated_at is not None
    assert payload.status == "queued"
    assert payload.progress == 0
    assert payload.original_filename == "clip.mov"
    assert payload.input_video_key == "inputs/job-model-1/input.mov"


def test_job_status_fields_update_cleanly(session_factory) -> None:
    _create_job(session_factory, id="job-update-1")

    with session_factory() as session:
        job = session.get(Job, "job-update-1")
        assert job is not None

        job.status = "running"
        job.progress = 45
        job.stage = "processing"
        job.message = "Processing frame batch."
        session.commit()
        session.refresh(job)

        assert job.status == "running"
        assert job.progress == 45
        assert job.stage == "processing"
        assert job.message == "Processing frame batch."


def test_create_job_uploads_file_persists_row_and_enqueues(
    client,
    session_factory,
    monkeypatch,
) -> None:
    storage_calls: list[dict[str, str | bytes]] = []
    enqueued_job_ids: list[str] = []

    class FakeStorage:
        def upload_input_file(self, job_id, upload) -> str:
            storage_calls.append(
                {
                    "job_id": job_id,
                    "filename": upload.filename or "",
                    "content_type": upload.content_type or "",
                    "content": upload.file.read(),
                }
            )
            return f"inputs/{job_id}/input.mp4"

    def fake_enqueue(job_id: str) -> str:
        enqueued_job_ids.append(job_id)
        return f"process-{job_id}"

    app.dependency_overrides[get_input_storage] = lambda: FakeStorage()
    monkeypatch.setattr(jobs_api, "enqueue_job", fake_enqueue)

    response = client.post(
        "/api/jobs",
        files={"file": ("receipt.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 201
    body = response.json()
    UUID(body["id"])
    assert body["status"] == "queued"
    assert body["original_filename"] == "receipt.mp4"
    assert body["input_video_key"] == f"inputs/{body['id']}/input.mp4"
    assert storage_calls == [
        {
            "job_id": body["id"],
            "filename": "receipt.mp4",
            "content_type": "video/mp4",
            "content": b"fake-video",
        }
    ]
    assert enqueued_job_ids == [body["id"]]

    with session_factory() as session:
        job = session.get(Job, body["id"])
        assert job is not None
        assert job.status == "queued"
        assert job.original_filename == "receipt.mp4"
        assert job.input_video_key == body["input_video_key"]


def test_get_job_returns_existing_job(client, session_factory) -> None:
    job = _create_job(session_factory, id="job-read-1")

    response = client.get(f"/api/jobs/{job.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "job-read-1"
    assert body["status"] == "queued"
    assert body["original_filename"] == "input.mp4"


def test_get_job_returns_404_for_unknown_job(client) -> None:
    response = client.get("/api/jobs/missing-job")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found."}
