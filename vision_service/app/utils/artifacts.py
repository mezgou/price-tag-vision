from __future__ import annotations

import json
from typing import Any

from app.services.storage import ArtifactStorage


class ArtifactWriter:
    def __init__(self, storage: ArtifactStorage, job_id: str) -> None:
        self.storage = storage
        self.job_id = job_id
        self.output_prefix = f"outputs/{job_id}"

    @property
    def csv_key(self) -> str:
        return self.build_key("result.csv")

    @property
    def preview_key(self) -> str:
        return self.build_key("preview.json")

    @property
    def manifest_key(self) -> str:
        return self.build_key("debug/pipeline_manifest.json")

    def build_key(self, relative_path: str) -> str:
        normalized_path = relative_path.strip().lstrip("/")
        return f"{self.output_prefix}/{normalized_path}"

    def upload_bytes(self, relative_path: str, payload: bytes, content_type: str) -> str:
        key = self.build_key(relative_path)
        self.storage.upload_bytes(key, payload, content_type)
        return key

    def upload_text(
        self,
        relative_path: str,
        payload: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> str:
        return self.upload_bytes(relative_path, payload.encode("utf-8"), content_type)

    def upload_json(self, relative_path: str, payload: Any) -> str:
        return self.upload_bytes(
            relative_path,
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )
