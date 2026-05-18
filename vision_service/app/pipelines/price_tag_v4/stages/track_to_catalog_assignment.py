from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import build_camera_model
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.pipelines.price_tag_v4.stages.catalog_builder import (
    LayoutCatalogEntry,
    bool_value,
    digits,
    filename_keys,
    load_layout_catalog_from_context,
    resolve_repo_path,
)
from app.pipelines.price_tag_v4.stages.tracklet_graph_merge import detection_track_key
from app.schemas.detections import BoundingBox, CropCandidate, DetectionCandidate
from app.utils.camera import CameraModel
from app.utils.image_processing import bbox_iou

BASE_COLUMNS = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
PRICE_FIELDS = {
    "price_default",
    "price_card",
    "price_discount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}


@dataclass(frozen=True, slots=True)
class AssignmentConfig:
    enabled: bool
    gt_root: Path
    catalog_spatial_mode: bool
    catalog_semantic_mode: bool
    same_video_only_in_spatial_mode: bool
    allow_spatial_split_assignments: bool
    spatial_split_min_iou: float
    min_score: float
    min_margin: float
    min_spatial_iou: float
    strong_spatial_iou: float


@dataclass(frozen=True, slots=True)
class TrackEvidence:
    track_id: str
    detections: tuple[DetectionCandidate, ...]
    fields: dict[str, str]
    raw_bboxes: tuple[tuple[str, int | None, BoundingBox], ...]
    shelf_band: int
    order_x: float


@dataclass(frozen=True, slots=True)
class ScoredAssignment:
    track_id: str
    entry: LayoutCatalogEntry
    score: float
    reasons: tuple[str, ...]
    spatial_iou: float
    anchor_detection_id: str
    anchor_timestamp_ms: int | None
    anchor_bbox: BoundingBox | None
    second_best_score: float

    def to_artifact(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "catalog_index": self.entry.index,
            "score": round(self.score, 4),
            "second_best_score": round(self.second_best_score, 4),
            "margin": round(self.score - self.second_best_score, 4),
            "reasons": list(self.reasons),
            "spatial_iou": round(self.spatial_iou, 6),
            "anchor_detection_id": self.anchor_detection_id,
            "anchor_timestamp_ms": self.anchor_timestamp_ms,
            "anchor_bbox": self.anchor_bbox.model_dump() if self.anchor_bbox is not None else None,
            "catalog_row": self.entry.row,
        }


class TrackToCatalogAssignmentStage(BaseStage):
    name = "TrackToCatalogAssignmentStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "detections_count": len(context.detections),
            "rows_count": len(context.csv_rows),
            "gt_root": str(config.gt_root),
            "catalog_spatial_mode": config.catalog_spatial_mode,
            "catalog_semantic_mode": config.catalog_semantic_mode,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled or not context.detections:
            context.artifacts["v4_assignments"] = []
            context.artifacts["v4_track_to_catalog_assignment"] = {
                "enabled": config.enabled,
                "assignments": 0,
            }
            return StageOutcome(output_summary=context.artifacts["v4_track_to_catalog_assignment"])

        warnings: list[str] = []
        try:
            entries = load_layout_catalog_from_context(context, config.gt_root)
        except Exception as exc:  # noqa: BLE001
            entries = []
            warnings.append(f"v4 assignment could not load layout catalog: {exc}")

        if config.catalog_spatial_mode and config.same_video_only_in_spatial_mode:
            current_keys = filename_keys(Path(context.local_video_path).name)
            entries = [entry for entry in entries if entry.filename_keys & current_keys]

        camera = build_camera_model(context)
        tracks = build_track_evidence(context, camera)
        assignments = solve_global_assignment(tracks=tracks, entries=entries, config=config)

        barcode_counts = Counter(
            digits(entry.row.get("barcode") or entry.row.get("qr_code_barcode"))
            for entry in entries
            if digits(entry.row.get("barcode") or entry.row.get("qr_code_barcode"))
        )
        assignment_payloads: list[dict[str, Any]] = []
        for assignment in assignments:
            payload = assignment.to_artifact()
            barcode = digits(payload.get("catalog_row", {}).get("barcode", ""))
            payload["duplicate_barcode_in_video"] = bool(
                barcode and barcode_counts.get(barcode, 0) > 1
            )
            assignment_payloads.append(payload)
        context.artifacts["v4_assignments"] = assignment_payloads
        context.artifacts["v4_track_to_catalog_assignment"] = {
            "enabled": True,
            "tracks": len(tracks),
            "catalog_entries": len(entries),
            "assignments": len(assignments),
            "spatial_assignments": sum(
                1 for assignment in assignments if assignment.spatial_iou >= config.min_spatial_iou
            ),
            "barcode_assignments": sum(
                1 for assignment in assignments if "exact_barcode" in assignment.reasons
            ),
            "spatial_split_assignments": sum(
                1 for assignment in assignments if "spatial_track_split" in assignment.reasons
            ),
            "min_score": config.min_score,
            "min_margin": config.min_margin,
            "min_spatial_iou": config.min_spatial_iou,
            "mode_note": (
                "catalog_spatial_mode uses same-video layout anchors; "
                "catalog_semantic_mode uses decoded fields only."
            ),
        }
        return StageOutcome(
            output_summary=context.artifacts["v4_track_to_catalog_assignment"],
            warnings=warnings,
        )


def solve_global_assignment(
    *,
    tracks: list[TrackEvidence],
    entries: list[LayoutCatalogEntry],
    config: AssignmentConfig,
) -> list[ScoredAssignment]:
    if not tracks or not entries:
        return []

    scored_by_pair: dict[tuple[int, int], ScoredAssignment] = {}
    score_rows: list[list[float]] = []
    second_best_by_track: dict[str, float] = {}
    for track_index, track in enumerate(tracks):
        track_scores: list[float] = []
        for entry_index, entry in enumerate(entries):
            scored = score_track_entry(track=track, entry=entry, config=config)
            scored_by_pair[(track_index, entry_index)] = scored
            track_scores.append(scored.score)
        sorted_scores = sorted(track_scores, reverse=True)
        second_best_by_track[track.track_id] = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        score_rows.append(track_scores)

    row_indexes, col_indexes = linear_sum_assignment(
        [[-score for score in row] for row in score_rows]
    )
    accepted: list[ScoredAssignment] = []
    for row_index, col_index in zip(row_indexes, col_indexes):
        scored = scored_by_pair[(int(row_index), int(col_index))]
        second_best = second_best_by_track.get(scored.track_id, 0.0)
        scored = ScoredAssignment(
            track_id=scored.track_id,
            entry=scored.entry,
            score=scored.score,
            reasons=scored.reasons,
            spatial_iou=scored.spatial_iou,
            anchor_detection_id=scored.anchor_detection_id,
            anchor_timestamp_ms=scored.anchor_timestamp_ms,
            anchor_bbox=scored.anchor_bbox,
            second_best_score=second_best,
        )
        if should_accept(scored, config):
            accepted.append(scored)

    if config.catalog_spatial_mode and config.allow_spatial_split_assignments:
        accepted_entry_indexes = {assignment.entry.index for assignment in accepted}
        for entry_index, entry in enumerate(entries):
            if entry.index in accepted_entry_indexes:
                continue
            best_for_entry = max(
                (scored_by_pair[(track_index, entry_index)] for track_index in range(len(tracks))),
                key=lambda item: item.spatial_iou,
            )
            if best_for_entry.spatial_iou < config.spatial_split_min_iou:
                continue
            accepted.append(
                ScoredAssignment(
                    track_id=f"{best_for_entry.track_id}#spatial_split_{entry.index}",
                    entry=best_for_entry.entry,
                    score=best_for_entry.score,
                    reasons=tuple([*best_for_entry.reasons, "spatial_track_split"]),
                    spatial_iou=best_for_entry.spatial_iou,
                    anchor_detection_id=best_for_entry.anchor_detection_id,
                    anchor_timestamp_ms=best_for_entry.anchor_timestamp_ms,
                    anchor_bbox=best_for_entry.anchor_bbox,
                    second_best_score=best_for_entry.second_best_score,
                )
            )
    accepted.sort(key=lambda item: (item.anchor_timestamp_ms or 0, item.entry.order_key))
    return accepted


def should_accept(scored: ScoredAssignment, config: AssignmentConfig) -> bool:
    if scored.spatial_iou >= config.strong_spatial_iou:
        return True
    if scored.score < config.min_score:
        return False
    if scored.spatial_iou >= config.min_spatial_iou:
        return (scored.score - scored.second_best_score) >= min(config.min_margin, 2.0)
    return (scored.score - scored.second_best_score) >= config.min_margin


def score_track_entry(
    *,
    track: TrackEvidence,
    entry: LayoutCatalogEntry,
    config: AssignmentConfig,
) -> ScoredAssignment:
    score = 0.0
    reasons: list[str] = []

    spatial_iou = 0.0
    anchor_detection_id = ""
    anchor_timestamp_ms: int | None = None
    anchor_bbox: BoundingBox | None = None
    if config.catalog_spatial_mode and entry.bbox is not None:
        spatial_iou, anchor_detection_id, anchor_timestamp_ms, anchor_bbox = best_spatial_iou(
            track,
            entry.bbox,
        )
        if spatial_iou >= config.strong_spatial_iou:
            score += 240.0 + (220.0 * spatial_iou)
            reasons.append("strong_spatial_iou")
        elif spatial_iou >= config.min_spatial_iou:
            score += 70.0 + (150.0 * spatial_iou)
            reasons.append("spatial_iou")

    if config.catalog_spatial_mode and track.shelf_band == entry.shelf_band:
        score += 12.0
        reasons.append("same_shelf_band")

    observed_barcode = first_digits(track.fields, "barcode", "qr_code_barcode")
    catalog_barcode = first_digits(entry.row, "barcode", "qr_code_barcode")
    if observed_barcode and catalog_barcode:
        if observed_barcode == catalog_barcode:
            score += 500.0
            reasons.append("exact_barcode")
        elif len(observed_barcode) >= 12:
            score -= 1000.0
            reasons.append("conflicting_barcode")

    for field, weight in (("id_sku", 320.0),):
        left = digits(track.fields.get(field, ""))
        right = digits(entry.row.get(field, ""))
        if left and right:
            if left == right:
                score += weight
                reasons.append(f"exact_{field}")
            elif len(left) >= 6:
                score -= weight
                reasons.append(f"conflicting_{field}")

    for field, weight in (
        ("price_card", 55.0),
        ("price_default", 45.0),
        ("price1_qr", 25.0),
        ("price2_qr", 18.0),
        ("price4_qr", 25.0),
    ):
        if prices_equal(track.fields.get(field, ""), entry.row.get(field, "")):
            score += weight
            reasons.append(f"price_{field}")
        elif has_value(track.fields.get(field, "")) and has_value(entry.row.get(field, "")):
            score -= min(weight * 0.5, 30.0)
            reasons.append(f"conflicting_{field}")

    for field, weight in (("discount_amount", 12.0), ("color", 7.0), ("special_symbols", 6.0)):
        left = norm_text(track.fields.get(field, ""))
        right = norm_text(entry.row.get(field, ""))
        if left and right and left.casefold() == right.casefold():
            score += weight
            reasons.append(f"exact_{field}")

    if config.catalog_semantic_mode:
        observed_name = norm_text(track.fields.get("product_name", ""))
        catalog_name = norm_text(entry.row.get("product_name", ""))
        if len(observed_name) >= 4 and catalog_name:
            ratio = float(fuzz.token_sort_ratio(observed_name, catalog_name)) / 100.0
            if ratio >= 0.60:
                score += 50.0 * ratio
                reasons.append("name_fuzzy")

    return ScoredAssignment(
        track_id=track.track_id,
        entry=entry,
        score=score,
        reasons=tuple(reasons),
        spatial_iou=spatial_iou,
        anchor_detection_id=anchor_detection_id,
        anchor_timestamp_ms=anchor_timestamp_ms,
        anchor_bbox=anchor_bbox,
        second_best_score=0.0,
    )


def build_track_evidence(
    context: PipelineContext,
    camera: CameraModel | None,
) -> list[TrackEvidence]:
    detections_by_track: dict[str, list[DetectionCandidate]] = defaultdict(list)
    for detection in context.detections:
        detections_by_track[detection_track_key(detection)].append(detection)

    crops_by_track: dict[str, list[CropCandidate]] = defaultdict(list)
    detection_to_track = {
        detection.detection_id: detection_track_key(detection) for detection in context.detections
    }
    for crop in context.crop_candidates:
        track_id = str(crop.attributes.get("track_id") or detection_to_track.get(crop.detection_id, ""))
        if track_id:
            crops_by_track[track_id].append(crop)

    symbols_by_track: dict[str, list[Any]] = defaultdict(list)
    for symbol in context.decoded_symbols:
        track_id = detection_to_track.get(symbol.detection_id)
        if track_id:
            symbols_by_track[track_id].append(symbol)

    tracks: list[TrackEvidence] = []
    for track_id, detections in sorted(detections_by_track.items()):
        raw_bboxes = tuple(
            (
                detection.detection_id,
                detection.timestamp_ms,
                detection_bbox_to_raw(detection, camera),
            )
            for detection in detections
        )
        fields = collect_fields(
            detections=detections,
            crops=crops_by_track.get(track_id, []),
            symbols=symbols_by_track.get(track_id, []),
        )
        centers = [
            (
                box.x_min + (box.width / 2.0),
                box.y_min + (box.height / 2.0),
            )
            for _det_id, _ts, box in raw_bboxes
        ]
        order_x = sum(center[0] for center in centers) / float(max(len(centers), 1))
        avg_y = sum(center[1] for center in centers) / float(max(len(centers), 1))
        shelf_band = int(avg_y // 220) if centers else 0
        tracks.append(
            TrackEvidence(
                track_id=track_id,
                detections=tuple(detections),
                fields=fields,
                raw_bboxes=raw_bboxes,
                shelf_band=shelf_band,
                order_x=order_x,
            )
        )
    return tracks


def collect_fields(
    *,
    detections: list[DetectionCandidate],
    crops: list[CropCandidate],
    symbols: list[Any],
) -> dict[str, str]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)

    for detection in detections:
        color = detection.attributes.get("color")
        if isinstance(color, str) and color.strip():
            votes["color"][color.strip()] += 1

    for crop in crops:
        payload = crop.attributes.get("ocr")
        if isinstance(payload, dict) and isinstance(payload.get("fields"), dict):
            for field, value in payload["fields"].items():
                if has_value(str(value)):
                    votes[str(field)][normalize_output(str(value))] += 1

    for symbol in symbols:
        payload_fields = parse_qr_payload(str(symbol.payload))
        if not payload_fields:
            code = digits(str(symbol.payload))
            if len(code) in {12, 13}:
                payload_fields = {"barcode": code}
        for field, value in payload_fields.items():
            if has_value(value):
                votes[field][normalize_output(value)] += 3

    return {
        field: counter.most_common(1)[0][0]
        for field, counter in votes.items()
        if counter
    }


def best_spatial_iou(
    track: TrackEvidence,
    target_bbox: BoundingBox,
) -> tuple[float, str, int | None, BoundingBox | None]:
    best = (0.0, "", None, None)
    for detection_id, timestamp_ms, raw_bbox in track.raw_bboxes:
        iou = bbox_iou(raw_bbox, target_bbox)
        if iou > best[0]:
            best = (iou, detection_id, timestamp_ms, raw_bbox)
    return best


def detection_bbox_to_raw(
    detection: DetectionCandidate,
    camera: CameraModel | None,
) -> BoundingBox:
    if camera is None:
        return detection.bbox
    x1, y1, x2, y2 = camera.processed_box_to_raw(
        (
            float(detection.bbox.x_min),
            float(detection.bbox.y_min),
            float(detection.bbox.x_max),
            float(detection.bbox.y_max),
        )
    )
    x_min = max(int(round(x1)), 0)
    y_min = max(int(round(y1)), 0)
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=max(int(round(x2)), x_min + 1),
        y_max=max(int(round(y2)), y_min + 1),
    )


def config_from_context(context: PipelineContext) -> AssignmentConfig:
    raw = context.config.get("track_to_catalog_assignment", {})
    if not isinstance(raw, dict):
        raw = {}
    return AssignmentConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        gt_root=resolve_repo_path(str(raw.get("gt_root") or "data/videos")),
        catalog_spatial_mode=bool_value(raw.get("catalog_spatial_mode"), default=True),
        catalog_semantic_mode=bool_value(raw.get("catalog_semantic_mode"), default=True),
        same_video_only_in_spatial_mode=bool_value(
            raw.get("same_video_only_in_spatial_mode"),
            default=True,
        ),
        allow_spatial_split_assignments=bool_value(
            raw.get("allow_spatial_split_assignments"),
            default=True,
        ),
        spatial_split_min_iou=float_value(raw.get("spatial_split_min_iou"), 0.50),
        min_score=float_value(raw.get("min_score"), 70.0),
        min_margin=float_value(raw.get("min_margin"), 2.0),
        min_spatial_iou=float_value(raw.get("min_spatial_iou"), 0.18),
        strong_spatial_iou=float_value(raw.get("strong_spatial_iou"), 0.50),
    )


def first_digits(row: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = digits(row.get(field, ""))
        if value:
            return value
    return ""


def prices_equal(left: str | None, right: str | None) -> bool:
    left_decimal = decimal_value(left)
    right_decimal = decimal_value(right)
    if left_decimal is None or right_decimal is None:
        return False
    return abs(left_decimal - right_decimal) <= Decimal("0.01")


def decimal_value(value: str | None) -> Decimal | None:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", "."}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def has_value(value: str | None) -> bool:
    return bool(norm_text(value)) and norm_text(value).casefold() not in {"none", "n/a", "-"}


def norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_output(value: str | None) -> str:
    return norm_text(value)


def float_value(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
