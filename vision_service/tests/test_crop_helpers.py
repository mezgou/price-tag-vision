from __future__ import annotations

import cv2
import numpy as np

from app.schemas.detections import BoundingBox
from app.utils.image_processing import (
    clip_bbox_to_frame,
    compute_crop_quality,
    expand_and_clip_bbox,
)


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
