from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, SampledFrameMetadata, StageOutcome
from app.schemas.detections import BoundingBox, DetectionCandidate
from app.utils.drawing import draw_detection_candidates
from app.utils.image_processing import (
    ScoredBoundingBox,
    build_candidate_masks,
    non_max_suppress_boxes,
)

PRICE_TAG_CANDIDATE_LABEL = "price_tag_candidate"
DEFAULT_CANDIDATE_SOURCE = "heuristic_color_geometry_v1"


@dataclass(slots=True)
class CandidateDetectionConfig:
    enabled: bool
    source: str
    max_candidates_per_frame: int
    min_area_ratio: float
    max_area_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    nms_iou_threshold: float
    debug_save_overlays: bool
    debug_overlay_jpeg_quality: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "CandidateDetectionConfig":
        raw_config = context.config.get("candidate_detection", {})
        warnings: list[str] = []

        if not isinstance(raw_config, dict):
            warnings.append(
                "Candidate detection config must be a mapping. Falling back to defaults."
            )
            raw_config = {}

        return cls(
            enabled=_coerce_bool(
                raw_config.get("enabled"),
                default=True,
                field_name="enabled",
                warnings=warnings,
            ),
            source=_coerce_string(
                raw_config.get("source"),
                default=DEFAULT_CANDIDATE_SOURCE,
                field_name="source",
                warnings=warnings,
            ),
            max_candidates_per_frame=_coerce_positive_int(
                raw_config.get("max_candidates_per_frame"),
                default=50,
                field_name="max_candidates_per_frame",
                warnings=warnings,
            ),
            min_area_ratio=_coerce_ratio(
                raw_config.get("min_area_ratio"),
                default=0.00005,
                field_name="min_area_ratio",
                warnings=warnings,
            ),
            max_area_ratio=_coerce_ratio(
                raw_config.get("max_area_ratio"),
                default=0.08,
                field_name="max_area_ratio",
                warnings=warnings,
            ),
            min_aspect_ratio=_coerce_positive_float(
                raw_config.get("min_aspect_ratio"),
                default=1.2,
                field_name="min_aspect_ratio",
                warnings=warnings,
            ),
            max_aspect_ratio=_coerce_positive_float(
                raw_config.get("max_aspect_ratio"),
                default=8.0,
                field_name="max_aspect_ratio",
                warnings=warnings,
            ),
            nms_iou_threshold=_coerce_ratio(
                raw_config.get("nms_iou_threshold"),
                default=0.4,
                field_name="nms_iou_threshold",
                warnings=warnings,
            ),
            debug_save_overlays=_coerce_bool(
                raw_config.get("debug_save_overlays"),
                default=True,
                field_name="debug_save_overlays",
                warnings=warnings,
            ),
            debug_overlay_jpeg_quality=_coerce_jpeg_quality(
                raw_config.get("debug_overlay_jpeg_quality"),
                default=85,
                field_name="debug_overlay_jpeg_quality",
                warnings=warnings,
            ),
            warnings=warnings,
        )


class HeuristicCandidateDetectionStage(BaseStage):
    name = "HeuristicCandidateDetectionStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = CandidateDetectionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "sampled_frames_count": len(context.sampled_frames),
            "max_candidates_per_frame": config.max_candidates_per_frame,
            "source": config.source,
            "debug_save_overlays": config.debug_save_overlays,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = CandidateDetectionConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            context.detections = []
            context.debug_overlay_keys = []
            context.artifacts["debug_overlay_keys"] = []
            return StageOutcome(
                output_summary={
                    "enabled": False,
                    "frames_processed": len(context.sampled_frames),
                    "detections_count": 0,
                    "overlays_saved": 0,
                },
                warnings=warnings,
            )

        detections: list[DetectionCandidate] = []
        debug_overlay_keys: list[str] = []
        frames_with_candidates = 0

        for sequence_number, frame_meta in enumerate(context.sampled_frames, start=1):
            frame = _load_runtime_frame(frame_meta)
            if frame is None:
                warnings.append(
                    f"Sampled frame {frame_meta.frame_index} is missing a runtime frame reference and was skipped."
                )
                continue

            frame_candidates = _detect_candidates_for_frame(
                frame=frame,
                frame_meta=frame_meta,
                sequence_number=sequence_number,
                config=config,
            )
            if frame_candidates:
                frames_with_candidates += 1
            detections.extend(frame_candidates)

            if config.debug_save_overlays:
                overlay = draw_detection_candidates(frame, frame_candidates)
                overlay_key = _upload_overlay(
                    context=context,
                    frame=overlay,
                    sequence_number=sequence_number,
                    jpeg_quality=config.debug_overlay_jpeg_quality,
                )
                debug_overlay_keys.append(overlay_key)

        context.detections = detections
        context.debug_overlay_keys = debug_overlay_keys
        context.artifacts["debug_overlay_keys"] = list(debug_overlay_keys)

        return StageOutcome(
            output_summary={
                "enabled": True,
                "frames_processed": len(context.sampled_frames),
                "frames_with_candidates": frames_with_candidates,
                "detections_count": len(detections),
                "overlays_saved": len(debug_overlay_keys),
                "source": config.source,
            },
            warnings=warnings,
        )


def _load_runtime_frame(frame_meta: SampledFrameMetadata) -> np.ndarray | None:
    if frame_meta.local_frame_path is None or not frame_meta.local_frame_path.exists():
        return None
    return cv2.imread(str(frame_meta.local_frame_path))


def _detect_candidates_for_frame(
    *,
    frame: np.ndarray,
    frame_meta: SampledFrameMetadata,
    sequence_number: int,
    config: CandidateDetectionConfig,
) -> list[DetectionCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    if frame_area <= 0:
        return []

    white_mask, accent_mask, candidate_mask = build_candidate_masks(frame)
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_candidates: list[ScoredBoundingBox] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 1 or height <= 1:
            continue

        bbox_area = width * height
        area_ratio = bbox_area / float(frame_area)
        if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
            continue

        aspect_ratio = width / float(height)
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            continue

        rectangularity = contour_area / float(bbox_area)
        if rectangularity < 0.42:
            continue

        white_ratio = cv2.countNonZero(white_mask[y : y + height, x : x + width]) / float(bbox_area)
        accent_ratio = cv2.countNonZero(accent_mask[y : y + height, x : x + width]) / float(bbox_area)
        shelf_score = _shelf_score(y=y, height=height, frame_height=frame_height)
        aspect_score = _aspect_score(
            aspect_ratio=aspect_ratio,
            min_aspect_ratio=config.min_aspect_ratio,
            max_aspect_ratio=config.max_aspect_ratio,
        )
        confidence = _confidence_score(
            white_ratio=white_ratio,
            accent_ratio=accent_ratio,
            rectangularity=rectangularity,
            aspect_score=aspect_score,
            shelf_score=shelf_score,
        )

        raw_candidates.append(
            ScoredBoundingBox(
                bbox=BoundingBox(
                    x_min=x,
                    y_min=y,
                    x_max=x + width,
                    y_max=y + height,
                ),
                score=confidence,
                attributes={
                    "area_ratio": round(area_ratio, 6),
                    "aspect_ratio": round(aspect_ratio, 4),
                    "white_ratio": round(white_ratio, 4),
                    "accent_ratio": round(accent_ratio, 4),
                    "rectangularity": round(rectangularity, 4),
                    "shelf_score": round(shelf_score, 4),
                },
            )
        )

    filtered_candidates = non_max_suppress_boxes(
        raw_candidates,
        iou_threshold=config.nms_iou_threshold,
        max_candidates=config.max_candidates_per_frame,
    )

    detections: list[DetectionCandidate] = []
    for rank, candidate in enumerate(filtered_candidates, start=1):
        detections.append(
            DetectionCandidate(
                detection_id=(
                    f"frame_{frame_meta.frame_index:06d}_"
                    f"candidate_{sequence_number:03d}_{rank:02d}"
                ),
                frame_index=frame_meta.frame_index,
                timestamp_ms=frame_meta.timestamp_ms,
                label=PRICE_TAG_CANDIDATE_LABEL,
                bbox=candidate.bbox,
                confidence=round(candidate.score, 4),
                source=config.source,
                attributes={
                    **candidate.attributes,
                    "rank": rank,
                },
            )
        )

    return detections


def _confidence_score(
    *,
    white_ratio: float,
    accent_ratio: float,
    rectangularity: float,
    aspect_score: float,
    shelf_score: float,
) -> float:
    normalized_white = min(white_ratio / 0.55, 1.0)
    normalized_accent = min(accent_ratio / 0.12, 1.0)
    score = (
        0.38 * normalized_white
        + 0.17 * normalized_accent
        + 0.22 * min(rectangularity, 1.0)
        + 0.13 * aspect_score
        + 0.10 * shelf_score
    )
    return float(max(min(score, 1.0), 0.0))


def _aspect_score(
    *,
    aspect_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> float:
    preferred = min(max((min_aspect_ratio + max_aspect_ratio) / 3.0, 1.8), 3.5)
    distance = abs(aspect_ratio - preferred)
    scale = max(max_aspect_ratio - min_aspect_ratio, 1e-6)
    return max(1.0 - (distance / scale), 0.0)


def _shelf_score(*, y: int, height: int, frame_height: int) -> float:
    center_y = y + (height / 2.0)
    normalized_y = center_y / float(max(frame_height, 1))
    return max(min((normalized_y - 0.2) / 0.6, 1.0), 0.0)


def _upload_overlay(
    *,
    context: PipelineContext,
    frame: np.ndarray,
    sequence_number: int,
    jpeg_quality: int,
) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError(
            f"Failed to encode overlay frame {sequence_number} as JPEG."
        )
    return context.artifact_writer.upload_bytes(
        f"debug/overlays/frame_{sequence_number:06d}_detections.jpg",
        encoded.tobytes(),
        "image/jpeg",
    )


def _coerce_bool(
    value: Any,
    *,
    default: bool,
    field_name: str,
    warnings: list[str],
) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
    return default


def _coerce_string(
    value: Any,
    *,
    default: str,
    field_name: str,
    warnings: list[str],
) -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return value.strip()
    warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
    return default


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    field_name: str,
    warnings: list[str],
) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0:
        warnings.append(f"{field_name} must be > 0. Falling back to {default}.")
        return default
    return coerced


def _coerce_positive_float(
    value: Any,
    *,
    default: float,
    field_name: str,
    warnings: list[str],
) -> float:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0:
        warnings.append(f"{field_name} must be > 0. Falling back to {default}.")
        return default
    return coerced


def _coerce_ratio(
    value: Any,
    *,
    default: float,
    field_name: str,
    warnings: list[str],
) -> float:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0 or coerced > 1:
        warnings.append(
            f"{field_name} must be within (0, 1]. Falling back to {default}."
        )
        return default
    return coerced


def _coerce_jpeg_quality(
    value: Any,
    *,
    default: int,
    field_name: str,
    warnings: list[str],
) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced < 1 or coerced > 100:
        warnings.append(
            f"{field_name} must be between 1 and 100. Falling back to {default}."
        )
        return default
    return coerced
