from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from app.schemas.detections import BoundingBox, CropQuality

FrameArray = NDArray[np.uint8]


@dataclass(slots=True)
class ScoredBoundingBox:
    bbox: BoundingBox
    score: float
    attributes: dict[str, Any] = field(default_factory=dict)


def resize_to_max_width(frame: FrameArray, *, max_width: int) -> FrameArray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    resized_height = max(int(round(height * scale)), 1)
    return cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    inter_x_min = max(left.x_min, right.x_min)
    inter_y_min = max(left.y_min, right.y_min)
    inter_x_max = min(left.x_max, right.x_max)
    inter_y_max = min(left.y_max, right.y_max)

    inter_width = max(inter_x_max - inter_x_min, 0)
    inter_height = max(inter_y_max - inter_y_min, 0)
    if inter_width == 0 or inter_height == 0:
        return 0.0

    intersection = inter_width * inter_height
    union = left.area + right.area - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def clip_bbox_to_frame(
    bbox: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> BoundingBox | None:
    clipped_x_min = max(min(bbox.x_min, frame_width), 0)
    clipped_y_min = max(min(bbox.y_min, frame_height), 0)
    clipped_x_max = max(min(bbox.x_max, frame_width), 0)
    clipped_y_max = max(min(bbox.y_max, frame_height), 0)

    if clipped_x_max <= clipped_x_min or clipped_y_max <= clipped_y_min:
        return None

    return BoundingBox(
        x_min=clipped_x_min,
        y_min=clipped_y_min,
        x_max=clipped_x_max,
        y_max=clipped_y_max,
    )


def expand_and_clip_bbox(
    bbox: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> BoundingBox:
    padding_x = int(round(bbox.width * padding_ratio))
    padding_y = int(round(bbox.height * padding_ratio))
    expanded = BoundingBox(
        x_min=max(bbox.x_min - padding_x, 0),
        y_min=max(bbox.y_min - padding_y, 0),
        x_max=min(bbox.x_max + padding_x, frame_width),
        y_max=min(bbox.y_max + padding_y, frame_height),
    )
    return expanded


def non_max_suppress_boxes(
    candidates: Sequence[ScoredBoundingBox],
    *,
    iou_threshold: float,
    max_candidates: int | None = None,
) -> list[ScoredBoundingBox]:
    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[ScoredBoundingBox] = []

    for candidate in sorted_candidates:
        if any(bbox_iou(candidate.bbox, existing.bbox) > iou_threshold for existing in kept):
            continue

        kept.append(candidate)
        if max_candidates is not None and len(kept) >= max_candidates:
            break

    return kept


def build_candidate_masks(
    frame: FrameArray,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    yellow_mask = cv2.inRange(hsv, (12, 70, 120), (38, 255, 255))
    orange_mask = cv2.inRange(hsv, (4, 80, 120), (24, 255, 255))
    accent_mask = cv2.bitwise_or(yellow_mask, orange_mask)

    candidate_mask = cv2.bitwise_or(white_mask, cv2.dilate(accent_mask, np.ones((5, 5), np.uint8), iterations=1))
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 5), np.uint8),
        iterations=1,
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    return white_mask, accent_mask, candidate_mask


def compute_crop_quality(
    crop: FrameArray,
    *,
    frame_area: int,
) -> CropQuality:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    glare_ratio = float(np.mean(gray > 245))
    area_ratio = float((crop.shape[0] * crop.shape[1]) / float(max(frame_area, 1)))

    sharpness_score = min(sharpness / 500.0, 1.0)
    brightness_score = max(1.0 - (abs(brightness - 160.0) / 160.0), 0.0)
    contrast_score = min(contrast / 64.0, 1.0)
    glare_score = max(1.0 - min(glare_ratio / 0.25, 1.0), 0.0)
    area_score = min(area_ratio / 0.01, 1.0)

    score = (
        0.35 * sharpness_score
        + 0.20 * brightness_score
        + 0.15 * contrast_score
        + 0.20 * glare_score
        + 0.10 * area_score
    )

    return CropQuality(
        sharpness=round(sharpness, 4),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        glare_ratio=round(glare_ratio, 6),
        area_ratio=round(area_ratio, 6),
        score=round(float(max(min(score, 1.0), 0.0)), 6),
    )
