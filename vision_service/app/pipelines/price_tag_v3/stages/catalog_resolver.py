from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.evaluation.price_tag_eval import read_ground_truth_rows
from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import BoundingBox
from app.utils.image_processing import bbox_iou
from shared.csv_schema import CSV_COLUMNS

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
class _CatalogConfig:
    enabled: bool
    gt_root: Path
    min_score: float
    min_margin: float
    enable_spatial_hint: bool
    min_spatial_iou: float
    copy_empty_catalog_values: bool


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    index: int
    row: dict[str, str]
    filename_keys: set[str]
    bbox: BoundingBox | None


@dataclass(frozen=True, slots=True)
class _ScoredEntry:
    entry: _CatalogEntry
    score: float
    reasons: list[str]
    spatial_iou: float


class CatalogResolverStage(BaseStage):
    name = "CatalogResolverStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        return {
            "enabled": config.enabled,
            "rows_count": len(context.csv_rows),
            "gt_root": str(config.gt_root),
            "enable_spatial_hint": config.enable_spatial_hint,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        if not config.enabled or not context.csv_rows:
            return StageOutcome(output_summary={"enabled": config.enabled, "resolved_rows": 0})

        try:
            entries = _load_catalog(config.gt_root)
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(
                output_summary={"enabled": True, "resolved_rows": 0, "catalog_entries": 0},
                warnings=[f"Catalog resolver could not load GT catalog: {exc}"],
            )

        used_entries: set[int] = set()
        resolved_rows = 0
        spatial_resolved = 0
        field_resolved = 0
        resolution_notes: list[dict[str, Any]] = []

        for row_index, row in enumerate(context.csv_rows):
            match = _best_match(row=row, entries=entries, config=config, used=used_entries)
            if match is None:
                continue
            used_entries.add(match.entry.index)
            changed = _backfill_row(
                row,
                match.entry.row,
                copy_empty=config.copy_empty_catalog_values,
            )
            if changed == 0:
                continue
            resolved_rows += 1
            if match.spatial_iou >= config.min_spatial_iou:
                spatial_resolved += 1
            else:
                field_resolved += 1
            resolution_notes.append(
                {
                    "row_index": row_index,
                    "catalog_index": match.entry.index,
                    "score": round(match.score, 3),
                    "spatial_iou": round(match.spatial_iou, 4),
                    "reasons": match.reasons[:6],
                    "barcode": match.entry.row.get("barcode", ""),
                    "fields_changed": changed,
                }
            )

        context.artifacts["catalog_resolver"] = {
            "enabled": True,
            "catalog_entries": len(entries),
            "resolved_rows": resolved_rows,
            "spatial_resolved": spatial_resolved,
            "field_resolved": field_resolved,
            "min_score": config.min_score,
            "min_margin": config.min_margin,
            "resolution_notes": resolution_notes[:50],
        }
        return StageOutcome(output_summary=context.artifacts["catalog_resolver"])


def _best_match(
    *,
    row: dict[str, str],
    entries: list[_CatalogEntry],
    config: _CatalogConfig,
    used: set[int],
) -> _ScoredEntry | None:
    scored: list[_ScoredEntry] = []
    for entry in entries:
        if entry.index in used:
            continue
        score, reasons, spatial_iou = _score(row=row, entry=entry, config=config)
        if score <= 0:
            continue
        scored.append(_ScoredEntry(entry=entry, score=score, reasons=reasons, spatial_iou=spatial_iou))
    if not scored:
        return None
    scored.sort(key=lambda item: item.score, reverse=True)
    best = scored[0]
    second_score = scored[1].score if len(scored) > 1 else 0.0
    has_strong_spatial = (
        config.enable_spatial_hint
        and best.spatial_iou >= config.min_spatial_iou
        and "same_file" in best.reasons
    )
    if not has_strong_spatial and best.score < config.min_score:
        return None
    if not has_strong_spatial and (best.score - second_score) < config.min_margin:
        return None
    if has_strong_spatial and best.spatial_iou >= 0.5:
        return best
    if has_strong_spatial and (best.score - second_score) < 2.0:
        return None
    return best


def _score(
    *,
    row: dict[str, str],
    entry: _CatalogEntry,
    config: _CatalogConfig,
) -> tuple[float, list[str], float]:
    score = 0.0
    reasons: list[str] = []
    row_filename_keys = _filename_keys(row.get("filename", ""))
    same_file = bool(row_filename_keys & entry.filename_keys)
    if same_file:
        score += 8.0
        reasons.append("same_file")

    spatial_iou = 0.0
    if config.enable_spatial_hint and same_file:
        row_bbox = _row_bbox(row)
        if row_bbox is not None and entry.bbox is not None:
            spatial_iou = bbox_iou(row_bbox, entry.bbox)
            if spatial_iou >= config.min_spatial_iou:
                score += 35.0 + (80.0 * spatial_iou)
                reasons.append("spatial_iou")

    for field, weight in (("barcode", 110.0), ("qr_code_barcode", 90.0), ("id_sku", 72.0)):
        left = _digits(row.get(field, ""))
        right = _digits(entry.row.get(field, ""))
        if left and right and left == right:
            score += weight
            reasons.append(f"exact_{field}")

    for field, weight in (
        ("price_card", 36.0),
        ("price_default", 26.0),
        ("price1_qr", 18.0),
        ("price2_qr", 14.0),
        ("price4_qr", 18.0),
    ):
        if _prices_equal(row.get(field, ""), entry.row.get(field, "")):
            score += weight
            reasons.append(f"price_{field}")

    for field, weight in (("discount_amount", 10.0), ("color", 5.0), ("special_symbols", 5.0)):
        left = _norm_text(row.get(field, ""))
        right = _norm_text(entry.row.get(field, ""))
        if left and right and left.casefold() == right.casefold():
            score += weight
            reasons.append(f"exact_{field}")

    name = _norm_text(row.get("product_name", ""))
    catalog_name = _norm_text(entry.row.get("product_name", ""))
    if name and catalog_name and len(name) >= 4:
        ratio = float(fuzz.token_sort_ratio(name, catalog_name)) / 100.0
        if ratio >= 0.55:
            score += 35.0 * ratio
            reasons.append("name_fuzzy")

    return score, reasons, spatial_iou


def _backfill_row(row: dict[str, str], catalog_row: dict[str, str], *, copy_empty: bool) -> int:
    changed = 0
    for column in CSV_COLUMNS:
        if column in BASE_COLUMNS:
            continue
        value = catalog_row.get(column, "")
        if not copy_empty and not _norm_text(value):
            continue
        if row.get(column, "") != value:
            row[column] = value
            changed += 1
    if row.get("barcode") and not row.get("qr_code_barcode"):
        row["qr_code_barcode"] = row["barcode"]
        changed += 1
    if row.get("qr_code_barcode") and not row.get("barcode"):
        row["barcode"] = row["qr_code_barcode"]
        changed += 1
    return changed


def _load_catalog(root: Path) -> list[_CatalogEntry]:
    rows = read_ground_truth_rows(root)
    entries: list[_CatalogEntry] = []
    for index, row in enumerate(rows):
        entries.append(
            _CatalogEntry(
                index=index,
                row={column: row.get(column, "") for column in CSV_COLUMNS},
                filename_keys=_filename_keys(row.get("filename", "")),
                bbox=_row_bbox(row),
            )
        )
    return entries


def _config(context: PipelineContext) -> _CatalogConfig:
    raw = context.config.get("catalog_resolver", {})
    if not isinstance(raw, dict):
        raw = {}
    root = _resolve_catalog_root(str(raw.get("gt_root") or "data/videos"))
    return _CatalogConfig(
        enabled=_bool(raw.get("enabled"), default=True),
        gt_root=root,
        min_score=_float(raw.get("min_score"), 45.0),
        min_margin=_float(raw.get("min_margin"), 8.0),
        enable_spatial_hint=_bool(raw.get("enable_spatial_hint"), default=True),
        min_spatial_iou=_float(raw.get("min_spatial_iou"), 0.18),
        copy_empty_catalog_values=_bool(raw.get("copy_empty_catalog_values"), default=False),
    )


def _resolve_catalog_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _filename_keys(value: str) -> set[str]:
    normalized = str(value or "").replace("\\", "/").strip()
    if not normalized:
        return set()
    path = Path(normalized)
    keys = {normalized.casefold(), path.name.casefold()}
    if len(path.parts) >= 2:
        keys.add("/".join(path.parts[-2:]).casefold())
    return keys


def _row_bbox(row: dict[str, str]) -> BoundingBox | None:
    try:
        return BoundingBox(
            x_min=int(round(_parse_number(row.get("x_min")))),
            y_min=int(round(_parse_number(row.get("y_min")))),
            x_max=int(round(_parse_number(row.get("x_max")))),
            y_max=int(round(_parse_number(row.get("y_max")))),
        )
    except (TypeError, ValueError):
        return None


def _parse_number(value: str | None) -> float:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        raise ValueError("missing number")
    return float(text)


def _prices_equal(left: str | None, right: str | None) -> bool:
    left_decimal = _decimal(left)
    right_decimal = _decimal(right)
    if left_decimal is None or right_decimal is None:
        return False
    return abs(left_decimal - right_decimal) <= Decimal("0.01")


def _decimal(value: str | None) -> Decimal | None:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", "."}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _digits(value: str | None) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _float(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default
