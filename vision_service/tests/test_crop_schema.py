from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.detections import BoundingBox, CropCandidate, CropQuality


def test_crop_candidate_schema_accepts_valid_payload() -> None:
    crop = CropCandidate(
        crop_id="frame_000001_crop_000001",
        detection_id="frame_000001_candidate_001_01",
        frame_index=1,
        timestamp_ms=200,
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70),
        padded_bbox=BoundingBox(x_min=6, y_min=16, x_max=114, y_max=74),
        crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
        width=108,
        height=58,
        quality=CropQuality(
            sharpness=123.4,
            brightness=180.0,
            contrast=40.0,
            glare_ratio=0.02,
            area_ratio=0.01,
            score=0.74,
        ),
        source="crop_extraction_v1",
    )

    assert crop.width == 108
    assert crop.quality.score == 0.74


def test_crop_candidate_schema_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValidationError):
        CropCandidate(
            crop_id="crop",
            detection_id="det",
            frame_index=0,
            timestamp_ms=0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
            padded_bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
            crop_key="key",
            width=0,
            height=10,
            quality=CropQuality(
                sharpness=0.0,
                brightness=0.0,
                contrast=0.0,
                glare_ratio=0.0,
                area_ratio=0.0,
                score=0.0,
            ),
            source="crop_extraction_v1",
        )
