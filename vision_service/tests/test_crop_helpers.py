from __future__ import annotations

import cv2
import numpy as np

from app.schemas.detections import BoundingBox, CropCandidate
from app.utils.image_processing import (
    clip_bbox_to_frame,
    compute_crop_quality,
    expand_and_clip_bbox,
)
from app.utils.reporting import build_crop_quality_summary


def test_bbox_padding_and_clipping_helper_expands_within_frame() -> None:
    bbox = BoundingBox(x_min=10, y_min=15, x_max=50, y_max=35)

    clipped = clip_bbox_to_frame(bbox, frame_width=60, frame_height=40)
    padded = expand_and_clip_bbox(
        clipped,
        frame_width=60,
        frame_height=40,
        padding_ratio=0.2,
    )

    assert clipped == bbox
    assert padded.model_dump() == {
        "x_min": 2,
        "y_min": 11,
        "x_max": 58,
        "y_max": 39,
    }


def test_crop_quality_helper_distinguishes_sharp_blurred_and_glare_images() -> None:
    sharp = np.zeros((80, 160, 3), dtype=np.uint8)
    cv2.putText(
        sharp,
        "TAG",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    blurred = cv2.GaussianBlur(sharp, (11, 11), 0)
    glare = np.full((80, 160, 3), 250, dtype=np.uint8)
    cv2.rectangle(glare, (10, 20), (150, 60), (255, 255, 255), thickness=-1)

    sharp_quality = compute_crop_quality(sharp, frame_area=80 * 160 * 4)
    blurred_quality = compute_crop_quality(blurred, frame_area=80 * 160 * 4)
    glare_quality = compute_crop_quality(glare, frame_area=80 * 160 * 4)

    assert sharp_quality.sharpness > blurred_quality.sharpness
    assert sharp_quality.score > blurred_quality.score
    assert glare_quality.glare_ratio > sharp_quality.glare_ratio
    assert glare_quality.score < sharp_quality.score


def test_crop_quality_summary_handles_empty_and_non_empty_inputs() -> None:
    empty_summary = build_crop_quality_summary([])
    assert empty_summary == {
        "count": 0,
        "score_min": 0.0,
        "score_max": 0.0,
        "score_mean": 0.0,
        "score_median": 0.0,
        "sharpness_mean": 0.0,
        "glare_ratio_mean": 0.0,
    }

    quality_a = compute_crop_quality(np.full((40, 80, 3), 180, dtype=np.uint8), frame_area=40000)
    quality_b = compute_crop_quality(np.full((60, 100, 3), 220, dtype=np.uint8), frame_area=40000)
    crops = [
        CropCandidate(
            crop_id="crop_a",
            detection_id="det_a",
            frame_index=0,
            timestamp_ms=0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=80, y_max=40),
            padded_bbox=BoundingBox(x_min=0, y_min=0, x_max=80, y_max=40),
            crop_key="",
            width=80,
            height=40,
            quality=quality_a,
            source="crop_extraction_v1",
        ),
        CropCandidate(
            crop_id="crop_b",
            detection_id="det_b",
            frame_index=1,
            timestamp_ms=100,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60),
            padded_bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60),
            crop_key="",
            width=100,
            height=60,
            quality=quality_b,
            source="crop_extraction_v1",
        ),
    ]

    summary = build_crop_quality_summary(crops)

    assert summary["count"] == 2
    assert summary["score_min"] <= summary["score_max"]
    assert summary["score_mean"] >= 0.0
    assert summary["score_median"] >= 0.0
    assert summary["sharpness_mean"] >= 0.0
    assert summary["glare_ratio_mean"] >= 0.0
