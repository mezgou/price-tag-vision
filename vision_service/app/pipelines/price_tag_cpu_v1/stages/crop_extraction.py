from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, SampledFrameMetadata, StageOutcome
from app.schemas.detections import CropCandidate, DetectionCandidate
from app.utils.image_processing import (
    clip_bbox_to_frame,
    compute_crop_quality,
    expand_and_clip_bbox,
)

DEFAULT_CROP_SOURCE = "crop_extraction_v1"


@dataclass(slots=True)
class CropExtractionConfig:
    enabled: bool
    source: str
    bbox_padding_ratio: float
    min_crop_width: int
    min_crop_height: int
    max_crops_per_frame: int
    max_total_crops: int
    debug_save_crops: bool
    debug_crop_jpeg_quality: int
    top_crops_preview_limit: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "CropExtractionConfig":
        raw_config = context.config.get("crop_extraction", {})
        warnings: list[str] = []

        if not isinstance(raw_config, dict):
            warnings.append(
                "Crop extraction config must be a mapping. Falling back to defaults."
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
                default=DEFAULT_CROP_SOURCE,
                field_name="source",
                warnings=warnings,
            ),
            bbox_padding_ratio=_coerce_ratio_inclusive_zero(
                raw_config.get("bbox_padding_ratio"),
                default=0.08,
                field_name="bbox_padding_ratio",
                warnings=warnings,
            ),
            min_crop_width=_coerce_positive_int(
                raw_config.get("min_crop_width"),
                default=24,
                field_name="min_crop_width",
                warnings=warnings,
            ),
            min_crop_height=_coerce_positive_int(
                raw_config.get("min_crop_height"),
                default=16,
                field_name="min_crop_height",
                warnings=warnings,
            ),
            max_crops_per_frame=_coerce_positive_int(
                raw_config.get("max_crops_per_frame"),
                default=50,
                field_name="max_crops_per_frame",
                warnings=warnings,
            ),
            max_total_crops=_coerce_positive_int(
                raw_config.get("max_total_crops"),
                default=300,
                field_name="max_total_crops",
                warnings=warnings,
            ),
            debug_save_crops=_coerce_bool(
                raw_config.get("debug_save_crops"),
                default=True,
                field_name="debug_save_crops",
                warnings=warnings,
            ),
            debug_crop_jpeg_quality=_coerce_jpeg_quality(
                raw_config.get("debug_crop_jpeg_quality"),
                default=90,
                field_name="debug_crop_jpeg_quality",
                warnings=warnings,
            ),
            top_crops_preview_limit=_coerce_positive_int(
                raw_config.get("top_crops_preview_limit"),
                default=20,
                field_name="top_crops_preview_limit",
                warnings=warnings,
            ),
            warnings=warnings,
        )


class CropExtractionStage(BaseStage):
    name = "CropExtractionStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = CropExtractionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "detections_count": len(context.detections),
            "sampled_frames_count": len(context.sampled_frames),
            "max_crops_per_frame": config.max_crops_per_frame,
            "max_total_crops": config.max_total_crops,
            "bbox_padding_ratio": config.bbox_padding_ratio,
            "source": config.source,
            "debug_save_crops": config.debug_save_crops,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = CropExtractionConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            context.crop_candidates = []
            context.debug_crop_keys = []
            context.artifacts["debug_crop_keys"] = []
            return StageOutcome(
                output_summary={
                    "enabled": False,
                    "crops_count": 0,
                    "debug_crops_count": 0,
                    "skipped_crops_count": 0,
                },
                warnings=warnings,
            )

        if not context.detections:
            context.crop_candidates = []
            context.debug_crop_keys = []
            context.artifacts["debug_crop_keys"] = []
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "crops_count": 0,
                    "debug_crops_count": 0,
                    "skipped_crops_count": 0,
                },
                warnings=warnings,
            )

        frame_lookup = _build_frame_lookup(context.sampled_frames)
        frame_crop_counts: dict[int, int] = defaultdict(int)
        crop_candidates: list[CropCandidate] = []
        debug_crop_keys: list[str] = []
        skipped_crops_count = 0

        for detection in sorted(
            context.detections,
            key=lambda item: (item.frame_index, -item.confidence, item.detection_id),
        ):
            if len(crop_candidates) >= config.max_total_crops:
                skipped_crops_count += 1
                warnings.append(
                    "Crop extraction reached max_total_crops limit; remaining detections were skipped."
                )
                break

            frame_entry = frame_lookup.get(detection.frame_index)
            if frame_entry is None:
                skipped_crops_count += 1
                warnings.append(
                    f"Detection {detection.detection_id} references frame {detection.frame_index} without a runtime frame."
                )
                continue

            frame_meta, frame_sequence_number = frame_entry
            if frame_crop_counts[detection.frame_index] >= config.max_crops_per_frame:
                skipped_crops_count += 1
                continue

            frame = _load_runtime_frame(frame_meta)
            if frame is None:
                skipped_crops_count += 1
                warnings.append(
                    f"Runtime frame for detection {detection.detection_id} is missing or unreadable."
                )
                continue

            crop_candidate, crop_warning = _build_crop_candidate(
                context=context,
                frame=frame,
                frame_meta=frame_meta,
                frame_sequence_number=frame_sequence_number,
                detection=detection,
                crop_index=frame_crop_counts[detection.frame_index] + 1,
                config=config,
            )
            if crop_warning is not None:
                skipped_crops_count += 1
                warnings.append(crop_warning)
                continue

            assert crop_candidate is not None
            crop_candidates.append(crop_candidate)
            frame_crop_counts[detection.frame_index] += 1
            if config.debug_save_crops:
                debug_crop_keys.append(crop_candidate.crop_key)

        context.crop_candidates = crop_candidates
        context.debug_crop_keys = debug_crop_keys
        context.artifacts["debug_crop_keys"] = list(debug_crop_keys)

        return StageOutcome(
            output_summary={
                "enabled": True,
                "crops_count": len(crop_candidates),
                "debug_crops_count": len(debug_crop_keys),
                "skipped_crops_count": skipped_crops_count,
                "source": config.source,
            },
            warnings=warnings,
        )


def _build_frame_lookup(
    sampled_frames: list[SampledFrameMetadata],
) -> dict[int, tuple[SampledFrameMetadata, int]]:
    lookup: dict[int, tuple[SampledFrameMetadata, int]] = {}
    for sequence_number, frame_meta in enumerate(sampled_frames, start=1):
        lookup[frame_meta.frame_index] = (frame_meta, sequence_number)
    return lookup


def _load_runtime_frame(frame_meta: SampledFrameMetadata) -> np.ndarray | None:
    if frame_meta.local_frame_path is None or not frame_meta.local_frame_path.exists():
        return None
    return cv2.imread(str(frame_meta.local_frame_path))


def _build_crop_candidate(
    *,
    context: PipelineContext,
    frame: np.ndarray,
    frame_meta: SampledFrameMetadata,
    frame_sequence_number: int,
    detection: DetectionCandidate,
    crop_index: int,
    config: CropExtractionConfig,
) -> tuple[CropCandidate | None, str | None]:
    frame_height, frame_width = frame.shape[:2]
    clipped_bbox = clip_bbox_to_frame(
        detection.bbox,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if clipped_bbox is None:
        return None, (
            f"Detection {detection.detection_id} bbox is outside frame bounds after clipping."
        )

    padded_bbox = expand_and_clip_bbox(
        clipped_bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=config.bbox_padding_ratio,
    )

    crop = frame[
        padded_bbox.y_min : padded_bbox.y_max,
        padded_bbox.x_min : padded_bbox.x_max,
    ]
    if crop.size == 0:
        return None, (
            f"Detection {detection.detection_id} produced an empty crop after clipping."
        )

    crop_height, crop_width = crop.shape[:2]
    if crop_width < config.min_crop_width or crop_height < config.min_crop_height:
        return None, (
            f"Detection {detection.detection_id} crop is too small "
            f"({crop_width}x{crop_height}) after clipping/padding."
        )

    crop_key = ""
    if config.debug_save_crops:
        crop_key = _upload_crop(
            context=context,
            crop=crop,
            frame_sequence_number=frame_sequence_number,
            crop_index=crop_index,
            jpeg_quality=config.debug_crop_jpeg_quality,
        )

    quality = compute_crop_quality(
        crop,
        frame_area=frame_width * frame_height,
    )
    crop_id = f"frame_{frame_meta.frame_index:06d}_crop_{crop_index:06d}"

    return (
        CropCandidate(
            crop_id=crop_id,
            detection_id=detection.detection_id,
            frame_index=detection.frame_index,
            timestamp_ms=detection.timestamp_ms,
            bbox=clipped_bbox,
            padded_bbox=padded_bbox,
            crop_key=crop_key,
            width=crop_width,
            height=crop_height,
            quality=quality,
            source=config.source,
            attributes={
                "label": detection.label,
                "detection_confidence": detection.confidence,
            },
        ),
        None,
    )


def _upload_crop(
    *,
    context: PipelineContext,
    crop: np.ndarray,
    frame_sequence_number: int,
    crop_index: int,
    jpeg_quality: int,
) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError(
            f"Failed to encode crop for frame {frame_sequence_number} detection {crop_index}."
        )
    return context.artifact_writer.upload_bytes(
        (
            f"debug/crops/frame_{frame_sequence_number:06d}_"
            f"det_{crop_index:06d}.jpg"
        ),
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


def _coerce_ratio_inclusive_zero(
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
    if coerced < 0 or coerced > 1:
        warnings.append(
            f"{field_name} must be within [0, 1]. Falling back to {default}."
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
