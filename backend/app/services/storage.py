from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from fastapi import UploadFile

from app.core.config import get_settings


class InputStorage:
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )

    def upload_input_file(self, job_id: str, upload: UploadFile) -> str:
        key = f"inputs/{job_id}/input{self._file_extension(upload.filename)}"
        upload.file.seek(0)

        extra_args: dict[str, str] = {}
        if upload.content_type:
            extra_args["ContentType"] = upload.content_type

        if extra_args:
            self.client.upload_fileobj(
                upload.file,
                self.bucket,
                key,
                ExtraArgs=extra_args,
            )
        else:
            self.client.upload_fileobj(upload.file, self.bucket, key)

        return key

    def _file_extension(self, filename: str | None) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix and suffix[1:].isalnum():
            return suffix
        return ".bin"


@lru_cache(maxsize=1)
def get_input_storage() -> InputStorage:
    return InputStorage()
