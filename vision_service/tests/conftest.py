from __future__ import annotations
# ruff: noqa: E402

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

VISION_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_SERVICE_DIR.parent
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

existing_app_module = sys.modules.get("app")
if existing_app_module is not None:
    existing_app_path = Path(getattr(existing_app_module, "__file__", "")).resolve()
    if VISION_SERVICE_DIR not in existing_app_path.parents:
        for module_name in list(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                sys.modules.pop(module_name, None)

from app.pipelines.base import PipelineContext, VideoMetadata
from app.utils.artifacts import ArtifactWriter

os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("S3_ACCESS_KEY_ID", "test")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("S3_BUCKET", "artifacts")


class InMemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.downloads: dict[str, bytes] = {}

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.objects[key] = (payload, content_type)

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.downloads[key]
        destination.write_bytes(payload)

    def ensure_bucket(self) -> None:
        return None

    def ensure_object(self, key: str) -> None:
        if key not in self.downloads:
            raise KeyError(key)


@pytest.fixture()
def in_memory_storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture()
def pipeline_context(
    tmp_path: Path,
    in_memory_storage: InMemoryStorage,
) -> PipelineContext:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    local_video_path = input_dir / "input.mp4"
    local_video_path.write_bytes(b"")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_writer = ArtifactWriter(in_memory_storage, "job-123")

    return PipelineContext(
        job_id="job-123",
        input_video_key="inputs/job-123/input.mp4",
        pipeline_name="price_tag_cpu_v1",
        pipeline_version="0.1.0",
        config={"pipeline": {"name": "price_tag_cpu_v1", "version": "0.1.0"}},
        local_video_path=local_video_path,
        work_dir=tmp_path,
        output_dir=output_dir,
        artifact_writer=artifact_writer,
        video_metadata=VideoMetadata.from_input_key("inputs/job-123/input.mp4"),
        artifacts={
            "csv_key": artifact_writer.csv_key,
            "preview_key": artifact_writer.preview_key,
            "manifest_key": artifact_writer.manifest_key,
            "crop_keys": [],
            "debug_frame_keys": [],
            "debug_overlay_keys": [],
            "debug_crop_keys": [],
        },
    )


@pytest.fixture()
def app_client(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_storage: InMemoryStorage,
) -> TestClient:
    import app.main as app_main

    monkeypatch.setattr(app_main, "storage", in_memory_storage)

    with TestClient(app_main.app) as client:
        yield client
