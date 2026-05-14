from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.detections import BoundingBox, DetectionCandidate


def test_detection_candidate_schema_accepts_valid_payload() -> None:
    candidate = DetectionCandidate(
        detection_id="frame_000001_candidate_001_01",
        frame_index=1,
        timestamp_ms=200,
        label="price_tag_candidate",
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70),
        confidence=0.84,
        source="heuristic_color_geometry_v1",
        attributes={"rank": 1},
    )

    assert candidate.bbox.area == 5000
    assert candidate.confidence == 0.84
    assert candidate.attributes["rank"] == 1


def test_bounding_box_schema_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x_min=20, y_min=10, x_max=20, y_max=40)


def test_detection_candidate_schema_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        DetectionCandidate(
            detection_id="candidate",
            frame_index=0,
            timestamp_ms=0,
            label="price_tag_candidate",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
            confidence=1.2,
            source="heuristic_color_geometry_v1",
        )
