from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.core.config import get_settings


class StorageObjectNotFoundError(FileNotFoundError):
    pass


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
        self.presign_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_public_endpoint_url or settings.s3_endpoint_url,
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

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if self._is_not_found_error(exc):
                raise StorageObjectNotFoundError(key) from exc
            raise

        return response["Body"].read()

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self.presign_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
            },
            ExpiresIn=expires_in,
        )

    def object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if self._is_not_found_error(exc):
                return False
            raise

        return True

    def _file_extension(self, filename: str | None) -> str:
        suffix = Path(filename or "").suffix.lower()
        if suffix and suffix[1:].isalnum():
            return suffix
        return ".bin"

    def _is_not_found_error(self, exc: ClientError) -> bool:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        return error_code in {"404", "NoSuchKey", "NotFound"}


@lru_cache(maxsize=1)
def get_input_storage() -> InputStorage:
    return InputStorage()
