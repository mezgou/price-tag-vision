from __future__ import annotations

from typing import Any
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


class FakeStorage:
    def __init__(
        self,
        *,
        json_payload: Any | None = None,
        bytes_payload: bytes = b"",
        presigned_url_prefix: str = "http://localhost:9000/artifacts/",
    ) -> None:
        self.json_payload = json_payload
        self.bytes_payload = bytes_payload
        self.presigned_url_prefix = presigned_url_prefix
        self.json_keys: list[str] = []
        self.byte_keys: list[str] = []
        self.presigned_calls: list[tuple[str, int]] = []
        self.upload_calls: list[dict[str, str | bytes]] = []

    def upload_input_file(self, job_id, upload) -> str:
        self.upload_calls.append(
            {
                "job_id": job_id,
                "filename": upload.filename or "",
                "content_type": upload.content_type or "",
                "content": upload.file.read(),
            }
        )
        return f"inputs/{job_id}/input.mp4"

    def get_json(self, key: str) -> Any:
        self.json_keys.append(key)
        return self.json_payload

    def get_bytes(self, key: str) -> bytes:
        self.byte_keys.append(key)
        return self.bytes_payload

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        self.presigned_calls.append((key, expires_in))
        return f"{self.presigned_url_prefix}{key}?signature=test"


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
    assert payload.output_csv_key is None
    assert payload.preview_json_key is None
    assert payload.crop_keys_json is None
    assert payload.stats_json is None
    assert payload.pipeline_name == "mock"
    assert payload.pipeline_version == "0.1.0"


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
    enqueued_job_ids: list[str] = []
    storage = FakeStorage()

    def fake_enqueue(job_id: str) -> str:
        enqueued_job_ids.append(job_id)
        return f"process-{job_id}"

    app.dependency_overrides[get_input_storage] = lambda: storage
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
    assert body["pipeline_name"] == "mock"
    assert body["pipeline_version"] == "0.1.0"
    assert body["output_csv_key"] is None
    assert body["preview_json_key"] is None
    assert body["crop_keys_json"] is None
    assert body["stats_json"] is None
    assert storage.upload_calls == [
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


def test_get_preview_success(client, session_factory) -> None:
    job = _create_job(
        session_factory,
        id="job-preview-1",
        status="succeeded",
        progress=100,
        stage="completed",
        preview_json_key="outputs/job-preview-1/preview.json",
    )
    storage = FakeStorage(
        json_payload={
            "job_id": job.id,
            "columns": ["filename"],
            "rows": [{"filename": "input.mp4"}],
        }
    )
    app.dependency_overrides[get_input_storage] = lambda: storage

    response = client.get(f"/api/jobs/{job.id}/preview")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job.id,
        "columns": ["filename"],
        "rows": [{"filename": "input.mp4"}],
    }
    assert storage.json_keys == ["outputs/job-preview-1/preview.json"]


def test_get_preview_not_ready(client, session_factory) -> None:
    job = _create_job(
        session_factory,
        id="job-preview-pending",
        status="running",
        stage="calling_vision_service",
        preview_json_key="outputs/job-preview-pending/preview.json",
    )
    storage = FakeStorage(json_payload={"ok": True})
    app.dependency_overrides[get_input_storage] = lambda: storage

    response = client.get(f"/api/jobs/{job.id}/preview")

    assert response.status_code == 409
    assert response.json() == {"detail": "Job result is not ready."}
    assert storage.json_keys == []


def test_download_csv_success(client, session_factory) -> None:
    job = _create_job(
        session_factory,
        id="job-csv-1",
        status="succeeded",
        progress=100,
        stage="completed",
        output_csv_key="outputs/job-csv-1/result.csv",
    )
    storage = FakeStorage(bytes_payload=b"filename,price_default\ninput.mp4,199.99\n")
    app.dependency_overrides[get_input_storage] = lambda: storage

    response = client.get(f"/api/jobs/{job.id}/download/csv")

    assert response.status_code == 200
    assert response.content == b"filename,price_default\ninput.mp4,199.99\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="result_{job.id}.csv"'
    )
    assert storage.byte_keys == ["outputs/job-csv-1/result.csv"]


def test_download_csv_not_ready(client, session_factory) -> None:
    job = _create_job(
        session_factory,
        id="job-csv-pending",
        status="queued",
        output_csv_key="outputs/job-csv-pending/result.csv",
    )
    storage = FakeStorage(bytes_payload=b"ignored")
    app.dependency_overrides[get_input_storage] = lambda: storage

    response = client.get(f"/api/jobs/{job.id}/download/csv")

    assert response.status_code == 409
    assert response.json() == {"detail": "Job result is not ready."}
    assert storage.byte_keys == []


def test_get_crops_success(client, session_factory) -> None:
    job = _create_job(
        session_factory,
        id="job-crops-1",
        status="succeeded",
        progress=100,
        stage="completed",
        crop_keys_json=[
            "outputs/job-crops-1/crops/crop_001.jpg",
            "outputs/job-crops-1/crops/crop_002.jpg",
        ],
    )
    storage = FakeStorage()
    app.dependency_overrides[get_input_storage] = lambda: storage

    response = client.get(f"/api/jobs/{job.id}/crops")

    assert response.status_code == 200
    assert response.json() == [
        {
            "key": "outputs/job-crops-1/crops/crop_001.jpg",
            "url": (
                "http://localhost:9000/artifacts/"
                "outputs/job-crops-1/crops/crop_001.jpg?signature=test"
            ),
            "filename": "crop_001.jpg",
        },
        {
            "key": "outputs/job-crops-1/crops/crop_002.jpg",
            "url": (
                "http://localhost:9000/artifacts/"
                "outputs/job-crops-1/crops/crop_002.jpg?signature=test"
            ),
            "filename": "crop_002.jpg",
        },
    ]
    assert storage.presigned_calls == [
        ("outputs/job-crops-1/crops/crop_001.jpg", 3600),
        ("outputs/job-crops-1/crops/crop_002.jpg", 3600),
    ]


def test_get_result_unknown_job(client) -> None:
    for path in (
        "/api/jobs/missing-job/preview",
        "/api/jobs/missing-job/download/csv",
        "/api/jobs/missing-job/crops",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"detail": "Job not found."}
