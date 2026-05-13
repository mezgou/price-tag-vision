from __future__ import annotations

import httpx
from redis import Redis

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()

    with httpx.Client(timeout=3.0) as client:
        response = client.get(f"{settings.vision_service_url}/health")
        response.raise_for_status()


if __name__ == "__main__":
    main()
