from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import BoundingBox, DetectionCandidate
from app.utils.image_processing import bbox_iou


@dataclass(slots=True)
class SimpleTrackingConfig:
    enabled: bool
    iou_threshold: float
    center_distance_ratio: float
    max_time_gap_ms: int
    source: str
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "SimpleTrackingConfig":
        raw_config = context.config.get("tracking", {})
        warnings: list[str] = []
        if not isinstance(raw_config, dict):
            warnings.append("tracking config must be a mapping. Falling back to defaults.")
            raw_config = {}
        return cls(
            enabled=_coerce_bool(raw_config.get("enabled"), default=True),
            iou_threshold=_coerce_ratio(
                raw_config.get("iou_threshold"),
                default=0.18,
                field_name="tracking.iou_threshold",
                warnings=warnings,
            ),
            center_distance_ratio=_coerce_positive_float(
                raw_config.get("center_distance_ratio"),
                default=0.65,
                field_name="tracking.center_distance_ratio",
                warnings=warnings,
            ),
            max_time_gap_ms=_coerce_positive_int(
                raw_config.get("max_time_gap_ms"),
                default=2500,
                field_name="tracking.max_time_gap_ms",
                warnings=warnings,
            ),
            source=str(raw_config.get("source") or "simple_iou_center_tracker_v1"),
            warnings=warnings,
        )


@dataclass(slots=True)
class _TrackState:
    track_id: str
    last_detection: DetectionCandidate
    detections: list[DetectionCandidate] = field(default_factory=list)


class SimpleTrackingStage(BaseStage):
    name = "SimpleTrackingStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = SimpleTrackingConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "detections_count": len(context.detections),
            "iou_threshold": config.iou_threshold,
            "center_distance_ratio": config.center_distance_ratio,
            "max_time_gap_ms": config.max_time_gap_ms,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = SimpleTrackingConfig.from_context(context)
        warnings = list(config.warnings)
        if not config.enabled or not context.detections:
            return StageOutcome(
                output_summary={
                    "enabled": config.enabled,
                    "tracks_count": 0,
                    "detections_count": len(context.detections),
                },
                warnings=warnings,
            )

        tracks: list[_TrackState] = []
        for detection in sorted(
            context.detections,
            key=lambda item: (
                item.timestamp_ms if item.timestamp_ms is not None else item.frame_index,
                item.frame_index,
                -item.confidence,
            ),
        ):
            best_track: _TrackState | None = None
            best_score = 0.0
            for track in tracks:
                if not _within_time_gap(
                    left=track.last_detection,
                    right=detection,
                    max_time_gap_ms=config.max_time_gap_ms,
                ):
                    continue
                score = _match_score(
                    track.last_detection.bbox,
                    detection.bbox,
                    center_distance_ratio=config.center_distance_ratio,
                )
                if score > best_score:
                    best_track = track
                    best_score = score

            if best_track is None or best_score < config.iou_threshold:
                track_id = f"track_{len(tracks) + 1:05d}"
                detection.attributes["track_id"] = track_id
                detection.attributes["tracking_source"] = config.source
                tracks.append(
                    _TrackState(
                        track_id=track_id,
                        last_detection=detection,
                        detections=[detection],
                    )
                )
                continue

            detection.attributes["track_id"] = best_track.track_id
            detection.attributes["tracking_source"] = config.source
            detection.attributes["tracking_score"] = round(best_score, 6)
            best_track.last_detection = detection
            best_track.detections.append(detection)

        context.artifacts["tracks"] = [
            {
                "track_id": track.track_id,
                "detections_count": len(track.detections),
                "first_frame_index": track.detections[0].frame_index,
                "last_frame_index": track.detections[-1].frame_index,
            }
            for track in tracks
        ]
        return StageOutcome(
            output_summary={
                "enabled": True,
                "tracks_count": len(tracks),
                "detections_count": len(context.detections),
                "source": config.source,
            },
            warnings=warnings,
        )


def _within_time_gap(
    *,
    left: DetectionCandidate,
    right: DetectionCandidate,
    max_time_gap_ms: int,
) -> bool:
    if left.timestamp_ms is None or right.timestamp_ms is None:
        return right.frame_index >= left.frame_index
    return 0 <= right.timestamp_ms - left.timestamp_ms <= max_time_gap_ms


def _match_score(
    left: BoundingBox,
    right: BoundingBox,
    *,
    center_distance_ratio: float,
) -> float:
    iou_score = bbox_iou(left, right)
    left_center = ((left.x_min + left.x_max) / 2.0, (left.y_min + left.y_max) / 2.0)
    right_center = ((right.x_min + right.x_max) / 2.0, (right.y_min + right.y_max) / 2.0)
    dx = left_center[0] - right_center[0]
    dy = left_center[1] - right_center[1]
    distance = ((dx * dx) + (dy * dy)) ** 0.5
    scale = max(left.width, left.height, right.width, right.height, 1)
    center_score = max(1.0 - (distance / max(scale * center_distance_ratio, 1.0)), 0.0)
    return max(iou_score, center_score * 0.75)


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


def _coerce_positive_float(
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
    try:
        coerced = float(default if value is None else value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0 or coerced > 1:
        warnings.append(f"{field_name} must be within (0, 1]. Falling back to {default}.")
        return default
    return coerced
