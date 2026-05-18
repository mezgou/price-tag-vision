from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import BoundingBox, DetectionCandidate
from app.utils.image_processing import bbox_iou


@dataclass(frozen=True, slots=True)
class TrackletGraphMergeConfig:
    enabled: bool
    max_time_gap_ms: int
    min_edge_score: float
    max_y_center_delta_ratio: float
    max_center_distance_ratio: float


@dataclass(slots=True)
class TrackletSummary:
    track_id: str
    detections: list[DetectionCandidate]
    first_ms: int
    last_ms: int
    median_bbox: BoundingBox
    max_area_bbox: BoundingBox
    confidence: float


class TrackletGraphMergeStage(BaseStage):
    name = "TrackletGraphMergeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "detections_count": len(context.detections),
            "max_time_gap_ms": config.max_time_gap_ms,
            "min_edge_score": config.min_edge_score,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled or not context.detections:
            context.artifacts["v4_tracklet_graph_merge"] = {
                "enabled": config.enabled,
                "input_tracklets": 0,
                "merged_tracks": 0,
            }
            return StageOutcome(output_summary=context.artifacts["v4_tracklet_graph_merge"])

        summaries = build_tracklet_summaries(context.detections)
        parent = {summary.track_id: summary.track_id for summary in summaries}
        edges: list[dict[str, Any]] = []

        for left_index, left in enumerate(summaries):
            for right in summaries[left_index + 1 :]:
                score = edge_score(left, right, config)
                if score < config.min_edge_score:
                    continue
                union(parent, left.track_id, right.track_id)
                edges.append(
                    {
                        "a": left.track_id,
                        "b": right.track_id,
                        "score": round(score, 4),
                    }
                )

        component_members: dict[str, list[str]] = defaultdict(list)
        for summary in summaries:
            component_members[find(parent, summary.track_id)].append(summary.track_id)

        merged_id_by_source: dict[str, str] = {}
        merged_tracks: list[dict[str, Any]] = []
        for index, source_ids in enumerate(
            sorted(component_members.values(), key=lambda items: (min(items), len(items))),
            start=1,
        ):
            merged_id = f"merged_track_{index:05d}"
            for source_id in source_ids:
                merged_id_by_source[source_id] = merged_id
            component_detections = [
                detection
                for source_id in source_ids
                for detection in next(
                    item.detections for item in summaries if item.track_id == source_id
                )
            ]
            merged_tracks.append(
                {
                    "merged_track_id": merged_id,
                    "source_track_ids": sorted(source_ids),
                    "detections_count": len(component_detections),
                    "frame_range": [
                        min(detection.frame_index for detection in component_detections),
                        max(detection.frame_index for detection in component_detections),
                    ],
                    "timestamp_range_ms": [
                        min(timestamp_or_zero(detection.timestamp_ms) for detection in component_detections),
                        max(timestamp_or_zero(detection.timestamp_ms) for detection in component_detections),
                    ],
                    "confidence": round(
                        sum(detection.confidence for detection in component_detections)
                        / float(max(len(component_detections), 1)),
                        4,
                    ),
                }
            )

        for detection in context.detections:
            source_track_id = detection_track_key(detection)
            detection.attributes["source_track_id"] = source_track_id
            detection.attributes["track_id"] = merged_id_by_source.get(source_track_id, source_track_id)
            detection.attributes["tracklet_graph_merge"] = "v4"

        context.artifacts["v4_merged_tracks"] = merged_tracks
        context.artifacts["v4_tracklet_graph_merge"] = {
            "enabled": True,
            "input_tracklets": len(summaries),
            "merged_tracks": len(merged_tracks),
            "merged_components": sum(
                1 for item in merged_tracks if len(item["source_track_ids"]) > 1
            ),
            "edges": edges[:100],
        }
        return StageOutcome(output_summary=context.artifacts["v4_tracklet_graph_merge"])


def build_tracklet_summaries(detections: list[DetectionCandidate]) -> list[TrackletSummary]:
    grouped: dict[str, list[DetectionCandidate]] = defaultdict(list)
    for detection in detections:
        grouped[detection_track_key(detection)].append(detection)

    summaries: list[TrackletSummary] = []
    for track_id, track_detections in grouped.items():
        boxes = [detection.bbox for detection in track_detections]
        summaries.append(
            TrackletSummary(
                track_id=track_id,
                detections=sorted(track_detections, key=lambda item: item.frame_index),
                first_ms=min(timestamp_or_zero(item.timestamp_ms) for item in track_detections),
                last_ms=max(timestamp_or_zero(item.timestamp_ms) for item in track_detections),
                median_bbox=median_bbox(boxes),
                max_area_bbox=max(boxes, key=lambda box: box.area),
                confidence=sum(item.confidence for item in track_detections)
                / float(max(len(track_detections), 1)),
            )
        )
    return summaries


def edge_score(
    left: TrackletSummary,
    right: TrackletSummary,
    config: TrackletGraphMergeConfig,
) -> float:
    if has_time_overlap(left, right):
        return 0.0
    gap = max(right.first_ms - left.last_ms, left.first_ms - right.last_ms, 0)
    if gap > config.max_time_gap_ms:
        return 0.0

    left_center = center(left.median_bbox)
    right_center = center(right.median_bbox)
    avg_height = max((left.median_bbox.height + right.median_bbox.height) / 2.0, 1.0)
    avg_diag = max(
        (diag(left.median_bbox) + diag(right.median_bbox)) / 2.0,
        1.0,
    )
    y_ratio = abs(left_center[1] - right_center[1]) / avg_height
    center_ratio = distance(left_center, right_center) / avg_diag
    if y_ratio > config.max_y_center_delta_ratio:
        return 0.0
    if center_ratio > config.max_center_distance_ratio:
        return 0.0

    iou = bbox_iou(left.median_bbox, right.median_bbox)
    size_ratio = min(left.max_area_bbox.area, right.max_area_bbox.area) / float(
        max(left.max_area_bbox.area, right.max_area_bbox.area, 1)
    )
    gap_score = 1.0 - min(gap / float(max(config.max_time_gap_ms, 1)), 1.0)
    y_score = 1.0 - min(y_ratio / max(config.max_y_center_delta_ratio, 0.001), 1.0)
    center_score = 1.0 - min(
        center_ratio / max(config.max_center_distance_ratio, 0.001),
        1.0,
    )
    return (0.35 * iou) + (0.25 * y_score) + (0.20 * center_score) + (0.10 * size_ratio) + (
        0.10 * gap_score
    )


def config_from_context(context: PipelineContext) -> TrackletGraphMergeConfig:
    raw = context.config.get("tracklet_graph_merge", {})
    if not isinstance(raw, dict):
        raw = {}
    return TrackletGraphMergeConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        max_time_gap_ms=positive_int(raw.get("max_time_gap_ms"), 3500),
        min_edge_score=float_value(raw.get("min_edge_score"), 0.82),
        max_y_center_delta_ratio=float_value(raw.get("max_y_center_delta_ratio"), 0.35),
        max_center_distance_ratio=float_value(raw.get("max_center_distance_ratio"), 0.70),
    )


def detection_track_key(detection: DetectionCandidate) -> str:
    value = detection.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return detection.detection_id


def median_bbox(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x_min=int(round(median([box.x_min for box in boxes]))),
        y_min=int(round(median([box.y_min for box in boxes]))),
        x_max=int(round(median([box.x_max for box in boxes]))),
        y_max=int(round(median([box.y_max for box in boxes]))),
    )


def has_time_overlap(left: TrackletSummary, right: TrackletSummary) -> bool:
    return left.first_ms <= right.last_ms and right.first_ms <= left.last_ms


def center(box: BoundingBox) -> tuple[float, float]:
    return (box.x_min + (box.width / 2.0), box.y_min + (box.height / 2.0))


def diag(box: BoundingBox) -> float:
    return math.hypot(box.width, box.height)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def timestamp_or_zero(value: int | None) -> int:
    return int(value or 0)


def find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = find(parent, left)
    right_root = find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def positive_int(value: Any, default: int) -> int:
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def float_value(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
