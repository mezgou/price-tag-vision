from __future__ import annotations

from typing import Any


def process_job(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "accepted",
        "payload": payload,
    }
