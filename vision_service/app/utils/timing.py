from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter


def utc_now() -> datetime:
    return datetime.now(UTC)


def elapsed_ms(start: float, end: float | None = None) -> int:
    stop = perf_counter() if end is None else end
    return max(int((stop - start) * 1000), 0)
