from __future__ import annotations

from functools import lru_cache

from redis import Redis
from rq import Queue

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url)


def get_queue() -> Queue:
    settings = get_settings()
    return Queue(settings.queue_name, connection=get_redis_connection())


def enqueue_job(job_id: str) -> str:
    rq_job = get_queue().enqueue(
        "app.worker.tasks.process_job",
        job_id,
        job_id=f"process-{job_id}",
    )
    return rq_job.id
