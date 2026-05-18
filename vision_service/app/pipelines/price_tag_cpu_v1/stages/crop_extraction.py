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
from app.utils.reporting import build_crop_quality_summary

DEFAULT_CROP_SOURCE = "crop_extraction_v1"
CONTACT_SHEET_BACKGROUND = (248, 248, 248)
CONTACT_SHEET_TEXT = (28, 28, 28)
CONTACT_SHEET_ACCENT = (0, 190, 255)


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
    debug_save_contact_sheet: bool
    contact_sheet_top_n: int
    contact_sheet_thumb_width: int
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
                default=0.1,
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
                default=16,
                field_name="max_crops_per_frame",
                warnings=warnings,
            ),
            max_total_crops=_coerce_positive_int(
                raw_config.get("max_total_crops"),
                default=120,
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
            debug_save_contact_sheet=_coerce_bool(
                raw_config.get("debug_save_contact_sheet"),
                default=True,
                field_name="debug_save_contact_sheet",
                warnings=warnings,
            ),
            contact_sheet_top_n=_coerce_positive_int(
                raw_config.get("contact_sheet_top_n"),
                default=40,
                field_name="contact_sheet_top_n",
                warnings=warnings,
            ),
            contact_sheet_thumb_width=_coerce_positive_int(
                raw_config.get("contact_sheet_thumb_width"),
                default=180,
                field_name="contact_sheet_thumb_width",
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
            "debug_save_contact_sheet": config.debug_save_contact_sheet,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = CropExtractionConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            return _reset_crop_outputs(context=context, enabled=False, warnings=warnings)

        if not context.detections:
            return _reset_crop_outputs(context=context, enabled=True, warnings=warnings)

        frame_lookup = _build_frame_lookup(context.sampled_frames)
        frame_crop_counts: dict[int, int] = defaultdict(int)
        crop_candidates: list[CropCandidate] = []
        debug_crop_keys: list[str] = []
        debug_contact_sheet_keys: list[str] = []
        contact_sheet_items: list[tuple[CropCandidate, np.ndarray]] = []
        skipped_crops_count = 0
        max_total_limit_warning_emitted = False

        for detection in sorted(
            context.detections,
            key=lambda item: (item.frame_index, -item.confidence, item.detection_id),
        ):
            if len(crop_candidates) >= config.max_total_crops:
                skipped_crops_count += 1
                if not max_total_limit_warning_emitted:
                    warnings.append(
                        "Crop extraction reached max_total_crops limit; remaining detections were skipped."
                    )
                    max_total_limit_warning_emitted = True
                continue

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

            crop_candidate, crop_image, crop_warning = _build_crop_candidate(
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
            assert crop_image is not None
            crop_candidates.append(crop_candidate)
            frame_crop_counts[detection.frame_index] += 1
            contact_sheet_items.append((crop_candidate, crop_image))
            if config.debug_save_crops and crop_candidate.crop_key:
                debug_crop_keys.append(crop_candidate.crop_key)

        if config.debug_save_contact_sheet and contact_sheet_items:
            contact_sheet_key = _upload_contact_sheet(
                context=context,
                contact_sheet_items=contact_sheet_items,
                top_n=config.contact_sheet_top_n,
                thumb_width=config.contact_sheet_thumb_width,
                jpeg_quality=config.debug_crop_jpeg_quality,
            )
            debug_contact_sheet_keys.append(contact_sheet_key)

        context.crop_candidates = crop_candidates
        context.debug_crop_keys = debug_crop_keys
        context.debug_contact_sheet_keys = debug_contact_sheet_keys
        context.artifacts["debug_crop_keys"] = list(debug_crop_keys)
        context.artifacts["debug_contact_sheet_keys"] = list(debug_contact_sheet_keys)

        crop_quality_summary = build_crop_quality_summary(crop_candidates)
        return StageOutcome(
            output_summary={
                "enabled": True,
                "crops_count": len(crop_candidates),
                "debug_crops_count": len(debug_crop_keys),
                "debug_contact_sheets_count": len(debug_contact_sheet_keys),
                "skipped_crops_count": skipped_crops_count,
                "source": config.source,
                "contact_sheet_keys": list(debug_contact_sheet_keys),
                "crop_quality_summary": crop_quality_summary,
            },
            warnings=warnings,
        )


def _reset_crop_outputs(
    *,
    context: PipelineContext,
    enabled: bool,
    warnings: list[str],
) -> StageOutcome:
    context.crop_candidates = []
    context.debug_crop_keys = []
    context.debug_contact_sheet_keys = []
    context.artifacts["debug_crop_keys"] = []
    context.artifacts["debug_contact_sheet_keys"] = []
    return StageOutcome(
        output_summary={
            "enabled": enabled,
            "crops_count": 0,
            "debug_crops_count": 0,
            "debug_contact_sheets_count": 0,
            "skipped_crops_count": 0,
            "contact_sheet_keys": [],
            "crop_quality_summary": build_crop_quality_summary([]),
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
) -> tuple[CropCandidate | None, np.ndarray | None, str | None]:
    frame_height, frame_width = frame.shape[:2]
    clipped_bbox = clip_bbox_to_frame(
        detection.bbox,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if clipped_bbox is None:
        return None, None, (
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
        return None, None, (
            f"Detection {detection.detection_id} produced an empty crop after clipping."
        )

    crop_height, crop_width = crop.shape[:2]
    if crop_width < config.min_crop_width or crop_height < config.min_crop_height:
        return None, None, (
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

    quality = compute_crop_quality(crop, frame_area=frame_width * frame_height)
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
                **detection.attributes,
                "label": detection.label,
                "detection_confidence": detection.confidence,
                "detection_source": detection.source,
            },
        ),
        crop.copy(),
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
        f"debug/crops/frame_{frame_sequence_number:06d}_det_{crop_index:06d}.jpg",
        encoded.tobytes(),
        "image/jpeg",
    )


def _upload_contact_sheet(
    *,
    context: PipelineContext,
    contact_sheet_items: list[tuple[CropCandidate, np.ndarray]],
    top_n: int,
    thumb_width: int,
    jpeg_quality: int,
) -> str:
    top_items = sorted(
        contact_sheet_items,
        key=lambda item: item[0].quality.score,
        reverse=True,
    )[:top_n]
    sheet_image = _build_contact_sheet_image(top_items=top_items, thumb_width=thumb_width)
    success, encoded = cv2.imencode(
        ".jpg",
        sheet_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError("Failed to encode crop contact sheet as JPEG.")
    return context.artifact_writer.upload_bytes(
        "debug/contact_sheets/top_crops.jpg",
        encoded.tobytes(),
        "image/jpeg",
    )


def _build_contact_sheet_image(
    *,
    top_items: list[tuple[CropCandidate, np.ndarray]],
    thumb_width: int,
) -> np.ndarray:
    if not top_items:
        return np.full((64, 64, 3), CONTACT_SHEET_BACKGROUND, dtype=np.uint8)

    padding = 12
    thumb_height = max(int(round(thumb_width * 0.72)), 90)
    text_block_height = 42
    columns = min(4, max(1, int(np.ceil(np.sqrt(len(top_items))))))
    rows = int(np.ceil(len(top_items) / columns))
    cell_width = thumb_width + (padding * 2)
    cell_height = thumb_height + text_block_height + (padding * 2)
    canvas = np.full(
        (rows * cell_height, columns * cell_width, 3),
        CONTACT_SHEET_BACKGROUND,
        dtype=np.uint8,
    )

    for item_index, (crop_candidate, crop_image) in enumerate(top_items):
        row_index = item_index // columns
        column_index = item_index % columns
        origin_x = column_index * cell_width
        origin_y = row_index * cell_height

        thumbnail = _fit_image_to_box(
            image=crop_image,
            target_width=thumb_width,
            target_height=thumb_height,
        )
        text_origin_y = origin_y + padding + 14
        cv2.putText(
            canvas,
            f"#{item_index + 1:02d} f={crop_candidate.frame_index} d={crop_candidate.attributes.get('detection_confidence', 0.0):.2f}",
            (origin_x + padding, text_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CONTACT_SHEET_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"q={crop_candidate.quality.score:.2f} {crop_candidate.width}x{crop_candidate.height}",
            (origin_x + padding, text_origin_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CONTACT_SHEET_TEXT,
            1,
            cv2.LINE_AA,
        )

        thumb_origin_y = origin_y + padding + text_block_height
        thumb_origin_x = origin_x + padding
        canvas[
            thumb_origin_y : thumb_origin_y + thumb_height,
            thumb_origin_x : thumb_origin_x + thumb_width,
        ] = thumbnail
        cv2.rectangle(
            canvas,
            (thumb_origin_x, thumb_origin_y),
            (thumb_origin_x + thumb_width - 1, thumb_origin_y + thumb_height - 1),
            CONTACT_SHEET_ACCENT,
            1,
        )

    return canvas


def _fit_image_to_box(
    *,
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    canvas = np.full(
        (target_height, target_width, 3),
        CONTACT_SHEET_BACKGROUND,
        dtype=np.uint8,
    )
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return canvas

    scale = min(target_width / float(source_width), target_height / float(source_height))
    resized_width = max(int(round(source_width * scale)), 1)
    resized_height = max(int(round(source_height * scale)), 1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


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
        warnings.append(f"{field_name} must be within [0, 1]. Falling back to {default}.")
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
