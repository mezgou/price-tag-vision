from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings


class VisionServiceError(RuntimeError):
    pass


class VisionProcessResponse(BaseModel):
    job_id: str
    status: str
    csv_key: str
    preview_key: str
    crop_keys: list[str]
    stats: dict[str, Any]


def process_video_job(
    *,
    job_id: str,
    input_video_key: str,
    pipeline_name: str,
    pipeline_version: str,
    config: dict[str, Any] | None = None,
) -> VisionProcessResponse:
    settings = get_settings()
    payload = {
        "job_id": job_id,
        "input_video_key": input_video_key,
        "pipeline_name": pipeline_name,
        "pipeline_version": pipeline_version,
        "config": config or {},
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.vision_service_url}/api/pipeline/process",
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _extract_error_detail(exc.response)
        raise VisionServiceError(
            f"Vision service returned HTTP {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise VisionServiceError(f"Vision service request failed: {exc}") from exc

    try:
        result = VisionProcessResponse.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        raise VisionServiceError(
            "Vision service returned an invalid response."
        ) from exc

    if result.status != "succeeded":
        raise VisionServiceError(f"Vision service returned status '{result.status}'.")

    return result


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

    if response.text.strip():
        return response.text.strip()

    return "No error details were provided."
