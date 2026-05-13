from __future__ import annotations

import logging
import threading
import time

import httpx
from redis import Redis
from redis.exceptions import RedisError
from rq import Worker

from app.core.config import get_settings
import app.worker.tasks  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("price-tag-vision.worker")


def wait_for_dependencies(redis_client: Redis, vision_service_url: str) -> None:
    while True:
        try:
            redis_client.ping()
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{vision_service_url}/health")
                response.raise_for_status()
            LOGGER.info("Worker dependencies are ready.")
            return
        except (RedisError, httpx.HTTPError) as exc:
            LOGGER.info("Waiting for worker dependencies: %s", exc)
            time.sleep(2)


def publish_heartbeat(redis_client: Redis, heartbeat_key: str, ttl_seconds: int) -> None:
    while True:
        try:
            redis_client.set(heartbeat_key, str(time.time()), ex=ttl_seconds * 2)
        except RedisError as exc:
            LOGGER.warning("Failed to update worker heartbeat: %s", exc)
        time.sleep(5)


def main() -> None:
    settings = get_settings()
    redis_client = Redis.from_url(settings.redis_url)
    worker = Worker([settings.queue_name], connection=redis_client)
    heartbeat_key = f"{settings.worker_heartbeat_prefix}:{worker.name}"

    wait_for_dependencies(redis_client, settings.vision_service_url)

    heartbeat_thread = threading.Thread(
        target=publish_heartbeat,
        args=(
            redis_client,
            heartbeat_key,
            settings.worker_heartbeat_ttl_seconds,
        ),
        daemon=True,
    )
    heartbeat_thread.start()

    LOGGER.info(
        "Starting RQ worker '%s' for queue '%s'.",
        worker.name,
        settings.queue_name,
    )
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
