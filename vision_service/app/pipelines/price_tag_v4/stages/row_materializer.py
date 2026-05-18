from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v4.stages.catalog_builder import digits
from app.pipelines.price_tag_v4.stages.track_to_catalog_assignment import (
    build_track_evidence,
)
from shared.csv_schema import CSV_COLUMNS

NO_VALUE = "нет"
DEFAULT_ABSENT_FIELDS = {
    "color",
    "price_discount",
    "price3_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
}
BASE_COLUMNS = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}


@dataclass(frozen=True, slots=True)
class RowMaterializerConfig:
    enabled: bool
    output_unassigned: bool
    copy_empty_catalog_values: bool
    default_absent_value: str


class RowMaterializerStage(BaseStage):
    name = "RowMaterializerStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        assignments = context.artifacts.get("v4_assignments", [])
        return {
            "enabled": config.enabled,
            "output_unassigned": config.output_unassigned,
            "assignments": len(assignments) if isinstance(assignments, list) else 0,
            "preliminary_rows": len(context.csv_rows),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled:
            return StageOutcome(output_summary={"enabled": False, "rows_count": len(context.csv_rows)})

        raw_assignments = context.artifacts.get("v4_assignments", [])
        if not isinstance(raw_assignments, list):
            raw_assignments = []

        track_evidence = {
            track.track_id: track
            for track in build_track_evidence(
                context,
                camera=None,
            )
        }
        materialized: list[dict[str, str]] = []
        assigned_track_ids: set[str] = set()
        conflicts = 0

        for assignment in raw_assignments:
            if not isinstance(assignment, dict):
                continue
            track_id = str(assignment.get("track_id", ""))
            catalog_row = assignment.get("catalog_row")
            if not track_id or not isinstance(catalog_row, dict):
                continue
            assigned_track_ids.add(track_id)
            row, had_conflict = materialize_assigned_row(
                context=context,
                assignment=assignment,
                catalog_row={column: str(catalog_row.get(column, "")) for column in CSV_COLUMNS},
                track_fields=track_evidence.get(track_id).fields if track_id in track_evidence else {},
                config=config,
            )
            conflicts += int(had_conflict)
            materialized.append(row)

        if config.output_unassigned:
            materialized.extend(context.csv_rows)

        materialized.sort(
            key=lambda row: (
                safe_int(row.get("frame_timestamp")),
                safe_float(row.get("x_min")),
                safe_float(row.get("y_min")),
            )
        )
        context.csv_rows = materialized
        context.artifacts["v4_row_materializer"] = {
            "enabled": True,
            "assigned_rows": len(raw_assignments),
            "rows_count": len(context.csv_rows),
            "output_unassigned": config.output_unassigned,
            "validated_barcode_conflicts": conflicts,
            "assigned_track_ids": sorted(assigned_track_ids)[:100],
        }
        return StageOutcome(output_summary=context.artifacts["v4_row_materializer"])


def materialize_assigned_row(
    *,
    context: PipelineContext,
    assignment: dict[str, Any],
    catalog_row: dict[str, str],
    track_fields: dict[str, str],
    config: RowMaterializerConfig,
) -> tuple[dict[str, str], bool]:
    row = {column: "" for column in CSV_COLUMNS}
    row["filename"] = catalog_row.get("filename") or Path(context.local_video_path).name

    timestamp_ms = assignment.get("anchor_timestamp_ms")
    if timestamp_ms is not None:
        row["frame_timestamp"] = str(timestamp_ms)
    bbox = assignment.get("anchor_bbox")
    if isinstance(bbox, dict):
        for column in ("x_min", "y_min", "x_max", "y_max"):
            value = bbox.get(column)
            row[column] = "" if value is None else f"{float(value):.1f}"

    for column in CSV_COLUMNS:
        if column in BASE_COLUMNS:
            continue
        value = catalog_row.get(column, "")
        if not config.copy_empty_catalog_values and not has_value(value):
            continue
        row[column] = value

    observed_barcode = first_valid_barcode(track_fields)
    catalog_barcode = digits(catalog_row.get("barcode") or catalog_row.get("qr_code_barcode"))
    conflict = bool(observed_barcode and catalog_barcode and observed_barcode != catalog_barcode)
    if conflict and "exact_barcode" not in set(assignment.get("reasons", [])):
        row["barcode"] = observed_barcode
        row["qr_code_barcode"] = observed_barcode

    if (
        assignment.get("duplicate_barcode_in_video") is True
        and "exact_barcode" not in set(assignment.get("reasons", []))
    ):
        row["barcode"] = ""
        row["qr_code_barcode"] = ""

    if has_value(row.get("barcode")) and not has_value(row.get("qr_code_barcode")):
        row["qr_code_barcode"] = row["barcode"]
    if has_value(row.get("qr_code_barcode")) and not has_value(row.get("barcode")):
        row["barcode"] = row["qr_code_barcode"]

    for field in DEFAULT_ABSENT_FIELDS:
        if not has_value(row.get(field)):
            row[field] = config.default_absent_value

    return ({column: normalize_output(row.get(column, "")) for column in CSV_COLUMNS}, conflict)


def config_from_context(context: PipelineContext) -> RowMaterializerConfig:
    raw = context.config.get("row_materializer_v4", {})
    if not isinstance(raw, dict):
        raw = {}
    return RowMaterializerConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        output_unassigned=bool_value(raw.get("output_unassigned"), default=False),
        copy_empty_catalog_values=bool_value(raw.get("copy_empty_catalog_values"), default=False),
        default_absent_value=normalize_output(str(raw.get("default_absent_value") or NO_VALUE)),
    )


def first_valid_barcode(fields: dict[str, str]) -> str:
    for field in ("barcode", "qr_code_barcode"):
        value = digits(fields.get(field, ""))
        if len(value) == 13 and ean13_ok(value):
            return value
    return ""


def ean13_ok(value: str) -> bool:
    code = digits(value)
    if len(code) != 13:
        return False
    checksum = sum((1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(code[:12]))
    return (10 - checksum % 10) % 10 == int(code[12])


def has_value(value: str | None) -> bool:
    text = normalize_output(value)
    return bool(text) and text.casefold() not in {"none", "n/a", "-"}


def normalize_output(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def safe_int(value: str | None) -> int:
    try:
        return int(float(str(value or "0").replace(",", ".")))
    except ValueError:
        return 0


def safe_float(value: str | None) -> float:
    try:
        return float(str(value or "0").replace(",", "."))
    except ValueError:
        return 0.0


def bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
