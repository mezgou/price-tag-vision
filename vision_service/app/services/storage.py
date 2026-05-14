from __future__ import annotations

from pathlib import Path

import boto3
from botocore.config import Config

from app.core.config import Settings


class ArtifactStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )

    def ensure_bucket(self) -> None:
        self.client.head_bucket(Bucket=self.settings.s3_bucket)

    def ensure_object(self, key: str) -> None:
        self.client.head_object(Bucket=self.settings.s3_bucket, Key=key)

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(
            self.settings.s3_bucket,
            key,
            str(destination),
        )

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.settings.s3_bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
        )
