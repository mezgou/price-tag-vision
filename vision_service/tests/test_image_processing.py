from __future__ import annotations

import pytest

from app.schemas.detections import BoundingBox
from app.utils.image_processing import ScoredBoundingBox, bbox_iou, non_max_suppress_boxes


def test_bbox_iou_returns_expected_overlap_ratio() -> None:
    left = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
    right = BoundingBox(x_min=25, y_min=25, x_max=125, y_max=125)

    iou = bbox_iou(left, right)

    assert iou == pytest.approx(0.3913, rel=1e-3)


def test_non_max_suppress_boxes_keeps_best_non_overlapping_candidates() -> None:
    candidates = [
        ScoredBoundingBox(
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
            score=0.9,
        ),
        ScoredBoundingBox(
            bbox=BoundingBox(x_min=10, y_min=10, x_max=95, y_max=95),
            score=0.8,
        ),
        ScoredBoundingBox(
            bbox=BoundingBox(x_min=160, y_min=20, x_max=240, y_max=80),
            score=0.7,
        ),
    ]

    kept = non_max_suppress_boxes(
        candidates,
        iou_threshold=0.4,
        max_candidates=10,
    )

    assert len(kept) == 2
    assert kept[0].score == 0.9
    assert kept[1].score == 0.7
