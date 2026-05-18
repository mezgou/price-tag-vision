from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import build_camera_model
from app.pipelines.price_tag_v4.stages.catalog_builder import (
    bool_value,
    digits,
    filename_keys,
    load_layout_catalog,
    resolve_repo_path,
)
from app.pipelines.price_tag_v4.stages.track_to_catalog_assignment import (
    detection_bbox_to_raw,
)
from app.pipelines.price_tag_v4.stages.tracklet_graph_merge import detection_track_key
from app.utils.image_processing import bbox_iou


@dataclass(frozen=True, slots=True)
class CoverageOracleConfig:
    enabled: bool
    gt_root: Path
    detection_iou_threshold: float


class CoverageOracleDebuggerStage(BaseStage):
    name = "CoverageOracleDebuggerStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "gt_root": str(config.gt_root),
            "detections_count": len(context.detections),
            "assignments": len(context.artifacts.get("v4_assignments", []))
            if isinstance(context.artifacts.get("v4_assignments"), list)
            else 0,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled:
            context.artifacts["v4_coverage_oracle"] = {"enabled": False}
            return StageOutcome(output_summary={"enabled": False})

        warnings: list[str] = []
        current_keys = filename_keys(Path(context.local_video_path).name)
        try:
            entries = [
                entry
                for entry in load_layout_catalog(config.gt_root)
                if entry.filename_keys & current_keys
            ]
        except Exception as exc:  # noqa: BLE001
            entries = []
            warnings.append(f"coverage oracle could not load GT rows: {exc}")

        camera = build_camera_model(context)
        detections = [
            (
                detection.detection_id,
                detection_track_key(detection),
                detection.frame_index,
                detection.timestamp_ms,
                detection_bbox_to_raw(detection, camera),
            )
            for detection in context.detections
        ]
        assignments = context.artifacts.get("v4_assignments", [])
        if not isinstance(assignments, list):
            assignments = []
        assigned_by_barcode = {
            digits(item.get("catalog_row", {}).get("barcode", ""))
            for item in assignments
            if isinstance(item, dict) and isinstance(item.get("catalog_row"), dict)
        }

        gt_reports: list[dict[str, Any]] = []
        no_detection = 0
        detected_unassigned = 0
        assigned = 0
        duplicate_candidates = 0

        for entry in entries:
            barcode = digits(entry.row.get("barcode", ""))
            best_iou = 0.0
            best_detection: dict[str, Any] | None = None
            duplicate_track_ids: set[str] = set()
            if entry.bbox is not None:
                for detection_id, track_id, frame_index, timestamp_ms, raw_bbox in detections:
                    iou = bbox_iou(raw_bbox, entry.bbox)
                    if iou >= config.detection_iou_threshold:
                        duplicate_track_ids.add(track_id)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = {
                            "detection_id": detection_id,
                            "track_id": track_id,
                            "frame_index": frame_index,
                            "timestamp_ms": timestamp_ms,
                        }

            is_assigned = barcode in assigned_by_barcode if barcode else False
            assigned += int(is_assigned)
            no_detection += int(best_iou < config.detection_iou_threshold)
            detected_unassigned += int(best_iou >= config.detection_iou_threshold and not is_assigned)
            duplicate_candidates += int(len(duplicate_track_ids) > 1)
            gt_reports.append(
                {
                    "catalog_index": entry.index,
                    "barcode": barcode,
                    "product_name": entry.row.get("product_name", ""),
                    "best_detection_iou": round(best_iou, 6),
                    "best_detection": best_detection,
                    "assigned": is_assigned,
                    "duplicate_track_candidates": sorted(duplicate_track_ids),
                }
            )

        payload = {
            "enabled": True,
            "filename": Path(context.local_video_path).name,
            "gt_rows": len(entries),
            "detections": len(context.detections),
            "assignments": len(assignments),
            "assigned_gt_by_barcode": assigned,
            "gt_without_detection_iou": no_detection,
            "gt_detected_but_unassigned": detected_unassigned,
            "gt_with_duplicate_track_candidates": duplicate_candidates,
            "detection_iou_threshold": config.detection_iou_threshold,
            "gt_reports": gt_reports,
        }
        key = context.artifact_writer.upload_json("debug/v4_coverage_oracle.json", payload)
        context.artifacts["v4_coverage_oracle"] = {
            **{key_name: payload[key_name] for key_name in payload if key_name != "gt_reports"},
            "artifact_key": key,
            "sample_reports": gt_reports[:25],
        }
        return StageOutcome(
            output_summary=context.artifacts["v4_coverage_oracle"],
            warnings=warnings,
        )


def config_from_context(context: PipelineContext) -> CoverageOracleConfig:
    raw = context.config.get("coverage_oracle_debugger", {})
    if not isinstance(raw, dict):
        raw = {}
    return CoverageOracleConfig(
        enabled=bool_value(raw.get("enabled"), default=False),
        gt_root=resolve_repo_path(str(raw.get("gt_root") or "data/videos")),
        detection_iou_threshold=float_value(raw.get("detection_iou_threshold"), 0.5),
    )


def float_value(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
