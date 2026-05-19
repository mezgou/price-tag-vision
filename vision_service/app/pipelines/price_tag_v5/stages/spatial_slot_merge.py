from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import build_camera_model
from app.schemas.detections import BoundingBox, CropCandidate, DetectionCandidate
from app.utils.camera import CameraModel
from app.utils.image_processing import bbox_iou


@dataclass(frozen=True, slots=True)
class SpatialSlotMergeConfig:
    enabled: bool
    max_time_gap_ms: int
    min_edge_score: float
    min_iou: float
    max_center_distance_ratio: float
    max_y_center_delta_ratio: float
    min_size_ratio: float
    allow_time_overlap: bool
    absorb_untracked_overlaps: bool
    overlap_absorb_min_iou: float
    overlap_absorb_max_detections: int

    @classmethod
    def from_context(cls, context: PipelineContext) -> "SpatialSlotMergeConfig":
        raw = context.config.get("v5_spatial_slot_merge", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=_bool(raw.get("enabled"), default=True),
            max_time_gap_ms=_positive_int(raw.get("max_time_gap_ms"), 1600),
            min_edge_score=_float(raw.get("min_edge_score"), 0.72),
            min_iou=_float(raw.get("min_iou"), 0.10),
            max_center_distance_ratio=_float(
                raw.get("max_center_distance_ratio"),
                0.55,
            ),
            max_y_center_delta_ratio=_float(
                raw.get("max_y_center_delta_ratio"),
                0.35,
            ),
            min_size_ratio=_float(raw.get("min_size_ratio"), 0.45),
            allow_time_overlap=_bool(raw.get("allow_time_overlap"), default=False),
            absorb_untracked_overlaps=_bool(
                raw.get("absorb_untracked_overlaps"),
                default=True,
            ),
            overlap_absorb_min_iou=_float(raw.get("overlap_absorb_min_iou"), 0.55),
            overlap_absorb_max_detections=_positive_int(
                raw.get("overlap_absorb_max_detections"),
                2,
            ),
        )


@dataclass(frozen=True, slots=True)
class TrackSlotSummary:
    track_id: str
    detections: tuple[DetectionCandidate, ...]
    first_time: int
    last_time: int
    median_bbox: BoundingBox
    max_area_bbox: BoundingBox
    confidence: float
    evidence_source: str


@dataclass(frozen=True, slots=True)
class SlotEdge:
    left_track_id: str
    right_track_id: str
    score: float
    iou: float
    center_ratio: float
    y_ratio: float
    size_ratio: float
    gap_ms: int
    reason: str

    def to_artifact(self) -> dict[str, Any]:
        return {
            "left": self.left_track_id,
            "right": self.right_track_id,
            "score": round(self.score, 4),
            "iou": round(self.iou, 4),
            "center_ratio": round(self.center_ratio, 4),
            "y_ratio": round(self.y_ratio, 4),
            "size_ratio": round(self.size_ratio, 4),
            "gap_ms": self.gap_ms,
            "reason": self.reason,
        }


class V5SpatialSlotMergeStage(BaseStage):
    name = "V5SpatialSlotMergeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = SpatialSlotMergeConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "detections": len(context.detections),
            "crops": len(context.crop_candidates),
            "max_time_gap_ms": config.max_time_gap_ms,
            "min_edge_score": config.min_edge_score,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = SpatialSlotMergeConfig.from_context(context)
        if not config.enabled or not context.detections:
            context.artifacts["v5_spatial_slot_merge"] = {
                "enabled": config.enabled,
                "input_tracks": 0,
                "slots": 0,
                "merged_components": 0,
            }
            return StageOutcome(
                output_summary=context.artifacts["v5_spatial_slot_merge"]
            )

        camera = build_camera_model(context)
        input_track_count = len({detection_track_key(d) for d in context.detections})
        track_to_slot, slots, edges = build_spatial_slot_assignments(
            context.detections,
            config=config,
            camera=camera,
            crops=context.crop_candidates,
        )
        detection_to_slot: dict[str, str] = {}
        for detection in context.detections:
            source_track_id = detection_track_key(detection)
            slot_id = track_to_slot.get(source_track_id, source_track_id)
            detection_to_slot[detection.detection_id] = slot_id
            detection.attributes["source_track_id"] = source_track_id
            detection.attributes["track_id"] = slot_id
            detection.attributes["spatial_slot_merge"] = "v5"

        for crop in context.crop_candidates:
            source_track_id = str(
                crop.attributes.get("source_track_id")
                or crop.attributes.get("track_id")
                or ""
            ).strip()
            if not source_track_id:
                source_track_id = crop.detection_id
            slot_id = detection_to_slot.get(
                crop.detection_id,
                track_to_slot.get(source_track_id, source_track_id),
            )
            crop.attributes["source_track_id"] = source_track_id
            crop.attributes["track_id"] = slot_id
            crop.attributes["spatial_slot_merge"] = "v5"

        summary = {
            "enabled": True,
            "input_tracks": input_track_count,
            "slots": len(slots),
            "merged_components": sum(
                1 for slot in slots if len(slot["source_track_ids"]) > 1
            ),
            "edges": [edge.to_artifact() for edge in edges[:100]],
            "slot_preview": slots[:100],
        }
        context.artifacts["v5_spatial_slot_merge"] = summary
        return StageOutcome(output_summary=summary)


def build_spatial_slot_assignments(
    detections: list[DetectionCandidate],
    *,
    config: SpatialSlotMergeConfig,
    camera: CameraModel | None,
    crops: list[CropCandidate] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], list[SlotEdge]]:
    summaries = build_track_slot_summaries(detections, camera, crops=crops)
    if not summaries:
        return {}, [], []

    parent = {summary.track_id: summary.track_id for summary in summaries}
    edges: list[SlotEdge] = []
    for left_index, left in enumerate(summaries):
        for right in summaries[left_index + 1 :]:
            edge = score_slot_edge(left, right, config)
            if edge is None or edge.score < config.min_edge_score:
                continue
            union(parent, left.track_id, right.track_id)
            edges.append(edge)

    components: dict[str, list[TrackSlotSummary]] = defaultdict(list)
    for summary in summaries:
        components[find(parent, summary.track_id)].append(summary)

    track_to_slot: dict[str, str] = {}
    slot_artifacts: list[dict[str, Any]] = []
    ordered_components = sorted(
        components.values(),
        key=lambda items: (
            min(item.first_time for item in items),
            min(center(item.median_bbox)[1] for item in items),
            min(center(item.median_bbox)[0] for item in items),
            min(item.track_id for item in items),
        ),
    )
    for index, members in enumerate(ordered_components, start=1):
        slot_id = f"slot_{index:05d}"
        for member in members:
            track_to_slot[member.track_id] = slot_id
        all_detections = [
            detection for member in members for detection in member.detections
        ]
        boxes = [detection_bbox_to_raw(detection, camera) for detection in all_detections]
        med = median_bbox(boxes)
        slot_artifacts.append(
            {
                "slot_id": slot_id,
                "source_track_ids": sorted(member.track_id for member in members),
                "detections_count": len(all_detections),
                "timestamp_range_ms": [
                    min(timestamp_key(detection) for detection in all_detections),
                    max(timestamp_key(detection) for detection in all_detections),
                ],
                "median_raw_bbox": med.model_dump(),
                "evidence_sources": sorted(
                    {member.evidence_source for member in members}
                ),
            }
        )
    return track_to_slot, slot_artifacts, edges


def build_track_slot_summaries(
    detections: list[DetectionCandidate],
    camera: CameraModel | None,
    *,
    crops: list[CropCandidate] | None = None,
) -> list[TrackSlotSummary]:
    grouped: dict[str, list[DetectionCandidate]] = defaultdict(list)
    for detection in detections:
        grouped[detection_track_key(detection)].append(detection)

    detection_to_track = {
        detection.detection_id: detection_track_key(detection)
        for detection in detections
    }
    crops_by_track: dict[str, list[CropCandidate]] = defaultdict(list)
    for crop in crops or []:
        track_id = str(
            crop.attributes.get("track_id")
            or detection_to_track.get(crop.detection_id, "")
        ).strip()
        if track_id:
            crops_by_track[track_id].append(crop)

    summaries: list[TrackSlotSummary] = []
    for track_id, track_detections in grouped.items():
        ordered = tuple(sorted(track_detections, key=timestamp_key))
        track_crops = sorted(crops_by_track.get(track_id, []), key=crop_timestamp_key)
        if track_crops:
            boxes = [bbox_to_raw(crop.bbox, camera) for crop in track_crops]
            first_time = min(crop_timestamp_key(crop) for crop in track_crops)
            last_time = max(crop_timestamp_key(crop) for crop in track_crops)
            evidence_source = "crops"
        else:
            boxes = [detection_bbox_to_raw(detection, camera) for detection in ordered]
            first_time = min(timestamp_key(detection) for detection in ordered)
            last_time = max(timestamp_key(detection) for detection in ordered)
            evidence_source = "detections"
        summaries.append(
            TrackSlotSummary(
                track_id=track_id,
                detections=ordered,
                first_time=first_time,
                last_time=last_time,
                median_bbox=median_bbox(boxes),
                max_area_bbox=max(boxes, key=lambda box: box.area),
                confidence=sum(detection.confidence for detection in ordered)
                / float(max(len(ordered), 1)),
                evidence_source=evidence_source,
            )
        )
    return summaries


def score_slot_edge(
    left: TrackSlotSummary,
    right: TrackSlotSummary,
    config: SpatialSlotMergeConfig,
) -> SlotEdge | None:
    time_overlap = has_time_overlap(left, right)
    overlap_absorb = False
    if time_overlap and not config.allow_time_overlap:
        overlap_absorb = can_absorb_untracked_overlap(left, right, config)
    if time_overlap and not config.allow_time_overlap and not overlap_absorb:
        return None
    gap = max(right.first_time - left.last_time, left.first_time - right.last_time, 0)
    if gap > config.max_time_gap_ms:
        return None

    left_center = center(left.median_bbox)
    right_center = center(right.median_bbox)
    avg_height = max((left.median_bbox.height + right.median_bbox.height) / 2.0, 1.0)
    avg_diag = max((diag(left.median_bbox) + diag(right.median_bbox)) / 2.0, 1.0)
    y_ratio = abs(left_center[1] - right_center[1]) / avg_height
    center_ratio = distance(left_center, right_center) / avg_diag
    iou = bbox_iou(left.median_bbox, right.median_bbox)
    size_ratio = min(left.max_area_bbox.area, right.max_area_bbox.area) / float(
        max(left.max_area_bbox.area, right.max_area_bbox.area, 1)
    )

    if overlap_absorb and iou < config.overlap_absorb_min_iou:
        return None
    if y_ratio > config.max_y_center_delta_ratio:
        return None
    if size_ratio < config.min_size_ratio:
        return None
    if iou < config.min_iou and center_ratio > config.max_center_distance_ratio:
        return None

    iou_score = min(iou / max(config.min_iou, 0.001), 1.0)
    y_score = 1.0 - min(y_ratio / max(config.max_y_center_delta_ratio, 0.001), 1.0)
    center_score = 1.0 - min(
        center_ratio / max(config.max_center_distance_ratio, 0.001),
        1.0,
    )
    size_score = min(size_ratio, 1.0)
    gap_score = 1.0 - min(gap / float(max(config.max_time_gap_ms, 1)), 1.0)
    score = (
        (0.30 * iou_score)
        + (0.25 * center_score)
        + (0.20 * y_score)
        + (0.15 * size_score)
        + (0.10 * gap_score)
    )
    reason = "overlap_untracked_absorb" if overlap_absorb else "temporal_spatial"
    return SlotEdge(
        left_track_id=left.track_id,
        right_track_id=right.track_id,
        score=score,
        iou=iou,
        center_ratio=center_ratio,
        y_ratio=y_ratio,
        size_ratio=size_ratio,
        gap_ms=gap,
        reason=reason,
    )


def can_absorb_untracked_overlap(
    left: TrackSlotSummary,
    right: TrackSlotSummary,
    config: SpatialSlotMergeConfig,
) -> bool:
    left_untracked = left.track_id.startswith("untracked_")
    right_untracked = right.track_id.startswith("untracked_")
    if left_untracked == right_untracked:
        return False
    short = left if left_untracked else right
    return len(short.detections) <= config.overlap_absorb_max_detections


def detection_bbox_to_raw(
    detection: DetectionCandidate,
    camera: CameraModel | None,
) -> BoundingBox:
    return bbox_to_raw(detection.bbox, camera)


def bbox_to_raw(
    bbox: BoundingBox,
    camera: CameraModel | None,
) -> BoundingBox:
    if camera is None:
        return bbox
    x1, y1, x2, y2 = camera.processed_box_to_raw(
        (
            float(bbox.x_min),
            float(bbox.y_min),
            float(bbox.x_max),
            float(bbox.y_max),
        )
    )
    x_min = max(int(round(min(x1, x2))), 0)
    y_min = max(int(round(min(y1, y2))), 0)
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=max(int(round(max(x1, x2))), x_min + 1),
        y_max=max(int(round(max(y1, y2))), y_min + 1),
    )


def median_bbox(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x_min=int(round(median([box.x_min for box in boxes]))),
        y_min=int(round(median([box.y_min for box in boxes]))),
        x_max=int(round(median([box.x_max for box in boxes]))),
        y_max=int(round(median([box.y_max for box in boxes]))),
    )


def detection_track_key(detection: DetectionCandidate) -> str:
    value = detection.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return detection.detection_id


def timestamp_key(detection: DetectionCandidate) -> int:
    if detection.timestamp_ms is not None:
        return int(detection.timestamp_ms)
    return int(detection.frame_index)


def crop_timestamp_key(crop: CropCandidate) -> int:
    if crop.timestamp_ms is not None:
        return int(crop.timestamp_ms)
    return int(crop.frame_index)


def has_time_overlap(left: TrackSlotSummary, right: TrackSlotSummary) -> bool:
    return left.first_time <= right.last_time and right.first_time <= left.last_time


def center(box: BoundingBox) -> tuple[float, float]:
    return (box.x_min + (box.width / 2.0), box.y_min + (box.height / 2.0))


def diag(box: BoundingBox) -> float:
    return math.hypot(box.width, box.height)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


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


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _float(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
