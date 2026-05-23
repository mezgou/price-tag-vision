from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import CropCandidate


@dataclass(slots=True)
class TopKCropSelectionConfig:
    enabled: bool
    max_crops_per_track: int
    max_total_crops: int
    min_quality_score: float
    temporal_diversity_enabled: bool
    temporal_diversity_buckets: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "TopKCropSelectionConfig":
        raw_config = context.config.get("top_k_crop_selection", {})
        warnings: list[str] = []
        if not isinstance(raw_config, dict):
            warnings.append(
                "top_k_crop_selection config must be a mapping. Falling back to defaults."
            )
            raw_config = {}
        return cls(
            enabled=_coerce_bool(raw_config.get("enabled"), default=True),
            max_crops_per_track=_coerce_positive_int(
                raw_config.get("max_crops_per_track"),
                default=10,
                field_name="top_k_crop_selection.max_crops_per_track",
                warnings=warnings,
            ),
            max_total_crops=_coerce_positive_int(
                raw_config.get("max_total_crops"),
                default=180,
                field_name="top_k_crop_selection.max_total_crops",
                warnings=warnings,
            ),
            min_quality_score=_coerce_ratio_inclusive_zero(
                raw_config.get("min_quality_score"),
                default=0.05,
                field_name="top_k_crop_selection.min_quality_score",
                warnings=warnings,
            ),
            temporal_diversity_enabled=_coerce_bool(
                raw_config.get("temporal_diversity_enabled"),
                default=False,
            ),
            temporal_diversity_buckets=_coerce_positive_int(
                raw_config.get("temporal_diversity_buckets"),
                default=5,
                field_name="top_k_crop_selection.temporal_diversity_buckets",
                warnings=warnings,
            ),
            warnings=warnings,
        )


class TopKCropSelectionStage(BaseStage):
    name = "TopKCropSelectionStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = TopKCropSelectionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops_per_track": config.max_crops_per_track,
            "max_total_crops": config.max_total_crops,
            "min_quality_score": config.min_quality_score,
            "temporal_diversity_enabled": config.temporal_diversity_enabled,
            "temporal_diversity_buckets": config.temporal_diversity_buckets,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = TopKCropSelectionConfig.from_context(context)
        warnings = list(config.warnings)
        if not config.enabled or not context.crop_candidates:
            return StageOutcome(
                output_summary={
                    "enabled": config.enabled,
                    "input_crops_count": len(context.crop_candidates),
                    "selected_crops_count": len(context.crop_candidates),
                },
                warnings=warnings,
            )

        grouped: dict[str, list[CropCandidate]] = defaultdict(list)
        for crop in context.crop_candidates:
            grouped[_crop_track_key(crop)].append(crop)

        selected: list[CropCandidate] = []
        selected_by_track: dict[str, list[str]] = {}
        for track_key, crops in grouped.items():
            best = _select_track_crops(crops=crops, config=config)
            selected.extend(best)
            selected_by_track[track_key] = [crop.crop_id for crop in best]

        selected.sort(
            key=lambda item: (
                item.quality.score,
                item.attributes.get("detection_confidence", 0.0),
            ),
            reverse=True,
        )
        if len(selected) > config.max_total_crops:
            selected = selected[: config.max_total_crops]
            kept_ids = {crop.crop_id for crop in selected}
            selected_by_track = {
                track_key: [crop_id for crop_id in crop_ids if crop_id in kept_ids]
                for track_key, crop_ids in selected_by_track.items()
            }

        selected.sort(key=lambda item: (item.frame_index, item.crop_id))
        context.crop_candidates = selected
        context.artifacts["selected_crop_ids_by_track"] = selected_by_track

        return StageOutcome(
            output_summary={
                "enabled": True,
                "input_crops_count": sum(len(crops) for crops in grouped.values()),
                "tracks_count": len(grouped),
                "selected_crops_count": len(selected),
                "max_crops_per_track": config.max_crops_per_track,
                "max_total_crops": config.max_total_crops,
                "temporal_diversity_enabled": config.temporal_diversity_enabled,
                "temporal_diversity_buckets": config.temporal_diversity_buckets,
            },
            warnings=warnings,
        )


def _select_track_crops(
    *,
    crops: list[CropCandidate],
    config: TopKCropSelectionConfig,
) -> list[CropCandidate]:
    quality_sorted = [
        crop
        for crop in sorted(crops, key=_quality_sort_key, reverse=True)
        if crop.quality.score >= config.min_quality_score
    ]
    if (
        not config.temporal_diversity_enabled
        or config.temporal_diversity_buckets <= 1
        or len(quality_sorted) <= config.max_crops_per_track
    ):
        return quality_sorted[: config.max_crops_per_track]

    return _select_temporal_diverse_crops(
        quality_sorted,
        max_crops=config.max_crops_per_track,
        bucket_count=config.temporal_diversity_buckets,
    )


def _select_temporal_diverse_crops(
    quality_sorted: list[CropCandidate],
    *,
    max_crops: int,
    bucket_count: int,
) -> list[CropCandidate]:
    bucket_count = min(bucket_count, max_crops, len(quality_sorted))
    if bucket_count <= 1:
        return quality_sorted[:max_crops]

    timestamps = [_temporal_value(crop) for crop in quality_sorted]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    if max_timestamp <= min_timestamp:
        return quality_sorted[:max_crops]

    buckets: list[list[CropCandidate]] = [[] for _ in range(bucket_count)]
    for crop in quality_sorted:
        bucket_index = _temporal_bucket_index(
            timestamp=_temporal_value(crop),
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            bucket_count=bucket_count,
        )
        buckets[bucket_index].append(crop)

    selected: list[CropCandidate] = []
    selected_ids: set[str] = set()
    for bucket in buckets:
        if not bucket:
            continue
        crop = bucket[0]
        selected.append(crop)
        selected_ids.add(crop.crop_id)
        if len(selected) >= max_crops:
            break

    for crop in quality_sorted:
        if len(selected) >= max_crops:
            break
        if crop.crop_id in selected_ids:
            continue
        selected.append(crop)
        selected_ids.add(crop.crop_id)

    return sorted(selected, key=_quality_sort_key, reverse=True)


def _quality_sort_key(crop: CropCandidate) -> tuple[float, float]:
    return (
        crop.quality.score,
        float(crop.attributes.get("detection_confidence", 0.0) or 0.0),
    )


def _temporal_value(crop: CropCandidate) -> float:
    if crop.timestamp_ms is not None:
        return float(crop.timestamp_ms)
    return float(crop.frame_index)


def _temporal_bucket_index(
    *,
    timestamp: float,
    min_timestamp: float,
    max_timestamp: float,
    bucket_count: int,
) -> int:
    ratio = (timestamp - min_timestamp) / max(max_timestamp - min_timestamp, 1.0)
    return min(max(int(ratio * bucket_count), 0), bucket_count - 1)


def _crop_track_key(crop: CropCandidate) -> str:
    track_id = crop.attributes.get("track_id")
    if isinstance(track_id, str) and track_id.strip():
        return track_id.strip()
    return crop.detection_id


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    field_name: str,
    warnings: list[str],
) -> int:
    try:
        coerced = int(default if value is None else value)
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
    try:
        coerced = float(default if value is None else value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced < 0 or coerced > 1:
        warnings.append(f"{field_name} must be within [0, 1]. Falling back to {default}.")
        return default
    return coerced
