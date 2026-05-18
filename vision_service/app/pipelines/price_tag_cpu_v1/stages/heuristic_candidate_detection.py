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
DEFAULT_CANDIDATE_SOURCE = "heuristic_color_geometry_v2"
MIN_CONFIDENCE_THRESHOLD = 0.18


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
    debug_save_masks: bool
    debug_save_rejected: bool
    top_rejected_preview_limit: int
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
                default=30,
                field_name="max_candidates_per_frame",
                warnings=warnings,
            ),
            min_area_ratio=_coerce_ratio(
                raw_config.get("min_area_ratio"),
                default=0.00008,
                field_name="min_area_ratio",
                warnings=warnings,
            ),
            max_area_ratio=_coerce_ratio(
                raw_config.get("max_area_ratio"),
                default=0.12,
                field_name="max_area_ratio",
                warnings=warnings,
            ),
            min_aspect_ratio=_coerce_positive_float(
                raw_config.get("min_aspect_ratio"),
                default=1.5,
                field_name="min_aspect_ratio",
                warnings=warnings,
            ),
            max_aspect_ratio=_coerce_positive_float(
                raw_config.get("max_aspect_ratio"),
                default=9.5,
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
            debug_save_masks=_coerce_bool(
                raw_config.get("debug_save_masks"),
                default=True,
                field_name="debug_save_masks",
                warnings=warnings,
            ),
            debug_save_rejected=_coerce_bool(
                raw_config.get("debug_save_rejected"),
                default=False,
                field_name="debug_save_rejected",
                warnings=warnings,
            ),
            top_rejected_preview_limit=_coerce_positive_int(
                raw_config.get("top_rejected_preview_limit"),
                default=20,
                field_name="top_rejected_preview_limit",
                warnings=warnings,
            ),
            warnings=warnings,
        )


@dataclass(slots=True)
class CandidateFeatures:
    area_ratio: float
    aspect_ratio: float
    rectangularity: float
    white_ratio: float
    yellow_or_orange_ratio: float
    edge_density: float
    text_like_score: float
    center_y_ratio: float
    glare_ratio: float


@dataclass(slots=True)
class FrameDetectionResult:
    detections: list[DetectionCandidate]
    raw_contours_count: int
    candidates_before_nms_count: int
    filtered_out_count: int
    nms_rejected_count: int
    rejected_previews: list[dict[str, Any]]


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
            "debug_save_masks": config.debug_save_masks,
            "debug_save_rejected": config.debug_save_rejected,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = CandidateDetectionConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            context.detections = []
            context.debug_overlay_keys = []
            context.debug_mask_keys = []
            context.artifacts["debug_overlay_keys"] = []
            context.artifacts["debug_mask_keys"] = []
            return StageOutcome(
                output_summary={
                    "enabled": False,
                    "frames_processed": len(context.sampled_frames),
                    "detections_count": 0,
                    "overlays_saved": 0,
                    "masks_saved": 0,
                },
                warnings=warnings,
            )

        detections: list[DetectionCandidate] = []
        debug_overlay_keys: list[str] = []
        debug_mask_keys: list[str] = []
        rejected_previews: list[dict[str, Any]] = []
        frames_with_candidates = 0
        raw_contours_total = 0
        candidates_before_nms_total = 0
        filtered_out_total = 0
        nms_rejected_total = 0

        for sequence_number, frame_meta in enumerate(context.sampled_frames, start=1):
            frame = _load_runtime_frame(frame_meta)
            if frame is None:
                warnings.append(
                    f"Sampled frame {frame_meta.frame_index} is missing a runtime frame reference and was skipped."
                )
                continue

            frame_result = _detect_candidates_for_frame(
                frame=frame,
                frame_meta=frame_meta,
                sequence_number=sequence_number,
                config=config,
            )
            if frame_result.detections:
                frames_with_candidates += 1
            detections.extend(frame_result.detections)
            raw_contours_total += frame_result.raw_contours_count
            candidates_before_nms_total += frame_result.candidates_before_nms_count
            filtered_out_total += frame_result.filtered_out_count
            nms_rejected_total += frame_result.nms_rejected_count
            rejected_previews.extend(frame_result.rejected_previews)

            if config.debug_save_overlays:
                overlay = draw_detection_candidates(frame, frame_result.detections)
                overlay_key = _upload_overlay(
                    context=context,
                    frame=overlay,
                    sequence_number=sequence_number,
                    jpeg_quality=config.debug_overlay_jpeg_quality,
                )
                debug_overlay_keys.append(overlay_key)

            if config.debug_save_masks:
                white_mask, accent_mask, combined_mask = build_candidate_masks(frame)
                debug_mask_keys.extend(
                    _upload_masks(
                        context=context,
                        white_mask=white_mask,
                        accent_mask=accent_mask,
                        combined_mask=combined_mask,
                        sequence_number=sequence_number,
                        jpeg_quality=config.debug_overlay_jpeg_quality,
                    )
                )

        context.detections = detections
        context.debug_overlay_keys = debug_overlay_keys
        context.debug_mask_keys = debug_mask_keys
        context.artifacts["debug_overlay_keys"] = list(debug_overlay_keys)
        context.artifacts["debug_mask_keys"] = list(debug_mask_keys)

        output_summary: dict[str, Any] = {
            "enabled": True,
            "frames_processed": len(context.sampled_frames),
            "frames_with_candidates": frames_with_candidates,
            "detections_count": len(detections),
            "overlays_saved": len(debug_overlay_keys),
            "masks_saved": len(debug_mask_keys),
            "raw_contours_total": raw_contours_total,
            "candidates_before_nms_total": candidates_before_nms_total,
            "candidates_after_nms_total": len(detections),
            "filtered_out_total": filtered_out_total,
            "nms_rejected_total": nms_rejected_total,
            "rejected_total": filtered_out_total + nms_rejected_total,
            "source": config.source,
        }

        if config.debug_save_rejected and rejected_previews:
            output_summary["top_rejected_candidates"] = sorted(
                rejected_previews,
                key=lambda item: (item.get("score", 0.0), item.get("frame_index", 0)),
                reverse=True,
            )[: config.top_rejected_preview_limit]

        return StageOutcome(output_summary=output_summary, warnings=warnings)


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
) -> FrameDetectionResult:
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    if frame_area <= 0:
        return FrameDetectionResult([], 0, 0, 0, 0, [])

    white_mask, accent_mask, candidate_mask = build_candidate_masks(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray_frame, 60, 160)
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    raw_candidates: list[ScoredBoundingBox] = []
    filtered_out_count = 0
    rejected_previews: list[dict[str, Any]] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        bbox_area = width * height
        rejection_payload = {
            "frame_index": frame_meta.frame_index,
            "bbox": {
                "x_min": x,
                "y_min": y,
                "x_max": x + max(width, 1),
                "y_max": y + max(height, 1),
            },
            "score": 0.0,
        }

        if contour_area <= 0 or width <= 1 or height <= 1 or bbox_area <= 0:
            filtered_out_count += 1
            rejected_previews.append(
                {
                    **rejection_payload,
                    "reason": "degenerate_bbox",
                }
            )
            continue

        features = _extract_candidate_features(
            gray_frame=gray_frame,
            white_mask=white_mask,
            accent_mask=accent_mask,
            edge_map=edge_map,
            bbox=(x, y, width, height),
            contour_area=contour_area,
            frame_area=frame_area,
            frame_height=frame_height,
        )

        rejection_reason = _rejection_reason(features=features, config=config)
        if rejection_reason is not None:
            filtered_out_count += 1
            rejected_previews.append(
                {
                    **rejection_payload,
                    "reason": rejection_reason,
                    "score": round(
                        score_candidate_features(features=features, config=config)[0],
                        4,
                    ),
                    "attributes": _serialize_candidate_attributes(
                        features=features,
                        score=0.0,
                        score_components={},
                    ),
                }
            )
            continue

        confidence, score_components = score_candidate_features(
            features=features,
            config=config,
        )
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            filtered_out_count += 1
            rejected_previews.append(
                {
                    **rejection_payload,
                    "reason": "low_confidence",
                    "score": round(confidence, 4),
                    "attributes": _serialize_candidate_attributes(
                        features=features,
                        score=confidence,
                        score_components=score_components,
                    ),
                }
            )
            continue

        raw_candidates.append(
            ScoredBoundingBox(
                bbox=BoundingBox(
                    x_min=x,
                    y_min=y,
                    x_max=x + width,
                    y_max=y + height,
                ),
                score=confidence,
                attributes=_serialize_candidate_attributes(
                    features=features,
                    score=confidence,
                    score_components=score_components,
                ),
            )
        )

    filtered_candidates = non_max_suppress_boxes(
        raw_candidates,
        iou_threshold=config.nms_iou_threshold,
        max_candidates=config.max_candidates_per_frame,
    )
    kept_ids = {id(candidate) for candidate in filtered_candidates}
    nms_rejected_count = max(len(raw_candidates) - len(filtered_candidates), 0)

    if nms_rejected_count > 0:
        for candidate in raw_candidates:
            if id(candidate) in kept_ids:
                continue
            rejected_previews.append(
                {
                    "frame_index": frame_meta.frame_index,
                    "reason": "suppressed_by_nms",
                    "score": round(candidate.score, 4),
                    "bbox": candidate.bbox.model_dump(),
                    "attributes": dict(candidate.attributes),
                }
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

    return FrameDetectionResult(
        detections=detections,
        raw_contours_count=len(contours),
        candidates_before_nms_count=len(raw_candidates),
        filtered_out_count=filtered_out_count,
        nms_rejected_count=nms_rejected_count,
        rejected_previews=rejected_previews,
    )


def score_candidate_features(
    *,
    features: CandidateFeatures,
    config: CandidateDetectionConfig,
) -> tuple[float, dict[str, float]]:
    shape_score = _normalized_range_score(features.rectangularity, 0.45, 0.98)
    aspect_score = _preferred_range_score(
        features.aspect_ratio,
        preferred_min=2.2,
        preferred_max=6.2,
        hard_min=config.min_aspect_ratio,
        hard_max=config.max_aspect_ratio,
    )
    area_score = _preferred_range_score(
        features.area_ratio,
        preferred_min=max(config.min_area_ratio * 3.0, 0.0004),
        preferred_max=min(config.max_area_ratio * 0.35, 0.03),
        hard_min=config.min_area_ratio,
        hard_max=config.max_area_ratio,
    )
    white_score = _normalized_range_score(features.white_ratio, 0.18, 0.82)
    accent_score = _normalized_range_score(features.yellow_or_orange_ratio, 0.01, 0.16)
    edge_score = _normalized_range_score(features.edge_density, 0.02, 0.18)
    text_score = float(max(min(features.text_like_score, 1.0), 0.0))
    position_score = _preferred_range_score(
        features.center_y_ratio,
        preferred_min=0.35,
        preferred_max=0.88,
        hard_min=0.0,
        hard_max=1.0,
    )
    glare_penalty = _normalized_range_score(features.glare_ratio, 0.10, 0.70)

    positive_score = (
        0.22 * shape_score
        + 0.18 * aspect_score
        + 0.08 * area_score
        + 0.14 * white_score
        + 0.10 * accent_score
        + 0.12 * edge_score
        + 0.12 * text_score
        + 0.06 * position_score
    )
    confidence = max(min(positive_score - (0.10 * glare_penalty), 1.0), 0.0)

    score_components = {
        "shape": round(0.22 * shape_score, 6),
        "aspect": round(0.18 * aspect_score, 6),
        "area": round(0.08 * area_score, 6),
        "white": round(0.14 * white_score, 6),
        "yellow_or_orange": round(0.10 * accent_score, 6),
        "edge_density": round(0.12 * edge_score, 6),
        "text_like": round(0.12 * text_score, 6),
        "center_y": round(0.06 * position_score, 6),
        "glare_penalty": round(-(0.10 * glare_penalty), 6),
        "total": round(confidence, 6),
    }
    return float(confidence), score_components


def _extract_candidate_features(
    *,
    gray_frame: np.ndarray,
    white_mask: np.ndarray,
    accent_mask: np.ndarray,
    edge_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    contour_area: float,
    frame_area: int,
    frame_height: int,
) -> CandidateFeatures:
    x, y, width, height = bbox
    bbox_area = width * height
    gray_roi = gray_frame[y : y + height, x : x + width]

    white_ratio = cv2.countNonZero(white_mask[y : y + height, x : x + width]) / float(
        bbox_area
    )
    yellow_or_orange_ratio = cv2.countNonZero(
        accent_mask[y : y + height, x : x + width]
    ) / float(bbox_area)
    edge_density = cv2.countNonZero(edge_map[y : y + height, x : x + width]) / float(
        bbox_area
    )
    center_y_ratio = (y + (height / 2.0)) / float(max(frame_height, 1))
    glare_ratio = float(np.mean(gray_roi >= 245))

    return CandidateFeatures(
        area_ratio=bbox_area / float(max(frame_area, 1)),
        aspect_ratio=width / float(max(height, 1)),
        rectangularity=contour_area / float(max(bbox_area, 1)),
        white_ratio=white_ratio,
        yellow_or_orange_ratio=yellow_or_orange_ratio,
        edge_density=edge_density,
        text_like_score=_estimate_text_like_score(gray_roi, roi_area=bbox_area),
        center_y_ratio=center_y_ratio,
        glare_ratio=glare_ratio,
    )


def _estimate_text_like_score(gray_roi: np.ndarray, *, roi_area: int) -> float:
    if gray_roi.size == 0 or roi_area <= 0:
        return 0.0

    blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_u8 = cv2.convertScaleAbs(magnitude)
    _, binary = cv2.threshold(
        gradient_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    component_count_score = 0.0
    component_result = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if isinstance(component_result, tuple) and len(component_result) == 4:
        _, _, stats, _ = component_result
        plausible_components = 0
        max_component_area = max(int(roi_area * 0.03), 8)
        for component_index in range(1, len(stats)):
            component_area = int(stats[component_index, cv2.CC_STAT_AREA])
            component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            if component_area < 4 or component_area > max_component_area:
                continue
            if component_width <= 1 or component_height <= 1:
                continue
            plausible_components += 1
        component_count_score = min(plausible_components / 18.0, 1.0)

    stroke_density = cv2.countNonZero(binary) / float(max(roi_area, 1))
    stroke_score = _normalized_range_score(stroke_density, 0.015, 0.14)
    dark_ratio = float(np.mean(gray_roi < 150))
    dark_score = _normalized_range_score(dark_ratio, 0.03, 0.35)

    return float(
        max(
            min(
                (0.45 * stroke_score) + (0.35 * component_count_score) + (0.20 * dark_score),
                1.0,
            ),
            0.0,
        )
    )


def _rejection_reason(
    *,
    features: CandidateFeatures,
    config: CandidateDetectionConfig,
) -> str | None:
    if features.area_ratio < config.min_area_ratio:
        return "area_too_small"
    if features.area_ratio > config.max_area_ratio:
        return "area_too_large"
    if features.aspect_ratio < config.min_aspect_ratio:
        return "aspect_ratio_too_small"
    if features.aspect_ratio > config.max_aspect_ratio:
        return "aspect_ratio_too_large"
    if features.rectangularity < 0.32:
        return "low_rectangularity"
    return None


def _serialize_candidate_attributes(
    *,
    features: CandidateFeatures,
    score: float,
    score_components: dict[str, float],
) -> dict[str, Any]:
    return {
        "area_ratio": round(features.area_ratio, 6),
        "aspect_ratio": round(features.aspect_ratio, 4),
        "rectangularity": round(features.rectangularity, 4),
        "white_ratio": round(features.white_ratio, 4),
        "yellow_or_orange_ratio": round(features.yellow_or_orange_ratio, 4),
        "edge_density": round(features.edge_density, 4),
        "text_like_score": round(features.text_like_score, 4),
        "center_y_ratio": round(features.center_y_ratio, 4),
        "glare_ratio": round(features.glare_ratio, 4),
        "score_components": score_components,
        "heuristic_score": round(score, 6),
    }


def _preferred_range_score(
    value: float,
    *,
    preferred_min: float,
    preferred_max: float,
    hard_min: float,
    hard_max: float,
) -> float:
    if hard_max <= hard_min:
        return 0.0
    if preferred_max < preferred_min:
        preferred_min, preferred_max = preferred_max, preferred_min
    if value < hard_min or value > hard_max:
        return 0.0
    if preferred_min <= value <= preferred_max:
        return 1.0
    if value < preferred_min:
        return max((value - hard_min) / max(preferred_min - hard_min, 1e-6), 0.0)
    return max((hard_max - value) / max(hard_max - preferred_max, 1e-6), 0.0)


def _normalized_range_score(value: float, lower_bound: float, upper_bound: float) -> float:
    if upper_bound <= lower_bound:
        return 0.0
    return max(min((value - lower_bound) / (upper_bound - lower_bound), 1.0), 0.0)


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
        raise RuntimeError(f"Failed to encode overlay frame {sequence_number} as JPEG.")
    return context.artifact_writer.upload_bytes(
        f"debug/overlays/frame_{sequence_number:06d}_detections.jpg",
        encoded.tobytes(),
        "image/jpeg",
    )


def _upload_masks(
    *,
    context: PipelineContext,
    white_mask: np.ndarray,
    accent_mask: np.ndarray,
    combined_mask: np.ndarray,
    sequence_number: int,
    jpeg_quality: int,
) -> list[str]:
    artifact_specs = [
        ("white_mask", white_mask),
        ("accent_mask", accent_mask),
        ("combined_mask", combined_mask),
    ]
    keys: list[str] = []

    for suffix, mask in artifact_specs:
        success, encoded = cv2.imencode(
            ".jpg",
            mask,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not success:
            raise RuntimeError(
                f"Failed to encode candidate detection mask {suffix} for frame {sequence_number}."
            )
        keys.append(
            context.artifact_writer.upload_bytes(
                f"debug/masks/frame_{sequence_number:06d}_{suffix}.jpg",
                encoded.tobytes(),
                "image/jpeg",
            )
        )

    return keys


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
        warnings.append(f"{field_name} must be within (0, 1]. Falling back to {default}.")
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
