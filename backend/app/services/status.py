from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
import httpx
import psycopg
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import Settings


def build_payload(service_name: str, services: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "service": service_name,
        "status": "ok" if is_healthy(services) else "degraded",
        "checked_at": _timestamp(),
        "services": services,
    }


def is_healthy(services: dict[str, dict[str, Any]]) -> bool:
    return all(service["status"] == "ok" for service in services.values())


def build_core_status(settings: Settings) -> dict[str, dict[str, Any]]:
    return {
        "backend": _status("ok", "HTTP API is responding."),
        "postgres": check_postgres(settings),
        "redis": check_redis(settings),
        "minio": check_minio(settings),
    }


def build_system_status(settings: Settings) -> dict[str, dict[str, Any]]:
    services = build_core_status(settings)
    services["vision_service"] = check_vision_service(settings)
    services["worker"] = check_worker(settings)
    return services


def check_postgres(settings: Settings) -> dict[str, Any]:
    try:
        dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(dsn, connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    except Exception as exc:  # pragma: no cover - defensive runtime check
        return _status("error", f"PostgreSQL unavailable: {exc}")
    return _status("ok", "PostgreSQL accepted a test query.")


def check_redis(settings: Settings) -> dict[str, Any]:
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
    except RedisError as exc:  # pragma: no cover - defensive runtime check
        return _status("error", f"Redis unavailable: {exc}")
    return _status("ok", "Redis responded to PING.")


def check_minio(settings: Settings) -> dict[str, Any]:
    try:
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        client.head_bucket(Bucket=settings.s3_bucket)
    except (BotoCoreError, ClientError) as exc:  # pragma: no cover - defensive runtime check
        return _status("error", f"MinIO bucket check failed: {exc}")
    return _status("ok", f"Bucket '{settings.s3_bucket}' is available.")


def check_vision_service(settings: Settings) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{settings.vision_service_url}/health")
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - defensive runtime check
        return _status("error", f"Vision service unavailable: {exc}")
    return _status("ok", "Vision service healthcheck succeeded.")


def check_worker(settings: Settings) -> dict[str, Any]:
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    except RedisError as exc:  # pragma: no cover - defensive runtime check
        return _status("error", f"Worker heartbeat unavailable: {exc}")

    heartbeat_pattern = f"{settings.worker_heartbeat_prefix}:*"
    heartbeat_keys = list(redis_client.scan_iter(match=heartbeat_pattern))

    if not heartbeat_keys:
        return _status("stale", "No worker heartbeats found yet.")

    fresh_workers = 0
    stale_workers = 0

    for heartbeat_key in heartbeat_keys:
        heartbeat_raw = redis_client.get(heartbeat_key)
        if not heartbeat_raw:
            stale_workers += 1
            continue

        try:
            age_seconds = time.time() - float(heartbeat_raw)
        except ValueError:
            redis_client.delete(heartbeat_key)
            stale_workers += 1
            continue

        if age_seconds <= settings.worker_heartbeat_ttl_seconds:
            fresh_workers += 1
        else:
            redis_client.delete(heartbeat_key)
            stale_workers += 1

    if fresh_workers == 0:
        return _status(
            "stale",
            f"No live workers. stale heartbeats={stale_workers}.",
        )

    return _status(
        "ok",
        f"Live workers={fresh_workers}. stale heartbeats={stale_workers}.",
    )


def _status(state: str, detail: str) -> dict[str, Any]:
    return {
        "status": state,
        "detail": detail,
        "checked_at": _timestamp(),
    }


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
