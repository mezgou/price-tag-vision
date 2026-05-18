from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.schemas.detections import BoundingBox
from app.utils.image_processing import bbox_iou
from shared.csv_schema import CSV_COLUMNS

COORDINATE_COLUMNS = {"x_min", "y_min", "x_max", "y_max"}
IGNORED_FIELD_COLUMNS = {"filename", "frame_timestamp", *COORDINATE_COLUMNS}
TEXT_SIMILARITY_FIELDS = {"product_name", "code", "additional_info"}
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
DIGIT_ONLY_FIELDS = {"barcode", "id_sku", "qr_code_barcode"}
HEADER_ALIASES = {"wholesale_level_1_coun": "wholesale_level_1_count"}


@dataclass(frozen=True, slots=True)
class MatchedRow:
    pred_index: int
    gt_index: int
    iou: float
    field_matches: int
    field_total: int
    field_accuracy: float
    is_correct: bool


def evaluate_prediction_csv(
    *,
    pred_csv_path: Path,
    gt_root: Path,
    iou_threshold: float = 0.5,
    row_field_accuracy_threshold: float = 0.8,
) -> dict[str, Any]:
    pred_rows = read_price_tag_csv(pred_csv_path)
    gt_rows = read_ground_truth_rows(gt_root)
    return evaluate_rows(
        pred_rows=pred_rows,
        gt_rows=gt_rows,
        iou_threshold=iou_threshold,
        row_field_accuracy_threshold=row_field_accuracy_threshold,
    )


def evaluate_rows(
    *,
    pred_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    iou_threshold: float = 0.5,
    row_field_accuracy_threshold: float = 0.8,
) -> dict[str, Any]:
    pred_by_video = _group_by_video(pred_rows)
    gt_by_video = _group_by_video(gt_rows)
    matched_rows: list[MatchedRow] = []
    unmatched_predictions = 0
    unmatched_ground_truth = 0

    for filename in sorted(set(pred_by_video) | set(gt_by_video)):
        predictions = pred_by_video.get(filename, [])
        ground_truth = gt_by_video.get(filename, [])
        matched_gt_indexes: set[int] = set()

        # Primary match key is barcode (task spec section 6): a prediction
        # whose barcode (or qr_code_barcode) matches a GT barcode is paired
        # directly, regardless of frame/bbox. Spatial IoU is the fallback for
        # the rest. This mirrors the hidden metric instead of being IoU-only.
        def _bc(row: dict[str, str]) -> str:
            for key in ("barcode", "qr_code_barcode"):
                digits = re.sub(r"\D", "", str(row.get(key) or ""))
                if len(digits) in (12, 13):
                    return digits
            return ""

        gt_by_barcode: dict[str, int] = {}
        for gt_index, gt_row in ground_truth:
            bc = _bc(gt_row)
            if bc and bc not in gt_by_barcode:
                gt_by_barcode[bc] = gt_index
        gt_lookup = {gt_index: gt_row for gt_index, gt_row in ground_truth}

        for pred_index, pred_row in predictions:
            best_match: tuple[int, dict[str, str], float] | None = None

            pred_bc = _bc(pred_row)
            if pred_bc and pred_bc in gt_by_barcode:
                gi = gt_by_barcode[pred_bc]
                if gi not in matched_gt_indexes:
                    best_match = (gi, gt_lookup[gi], 1.0)

            if best_match is None:
                pred_box = _row_bbox(pred_row)
                if pred_box is None:
                    unmatched_predictions += 1
                    continue
                for gt_index, gt_row in ground_truth:
                    if gt_index in matched_gt_indexes:
                        continue
                    gt_box = _row_bbox(gt_row)
                    if gt_box is None:
                        continue
                    iou = bbox_iou(pred_box, gt_box)
                    if iou < iou_threshold:
                        continue
                    if best_match is None or iou > best_match[2]:
                        best_match = (gt_index, gt_row, iou)

            if best_match is None:
                unmatched_predictions += 1
                continue

            gt_index, gt_row, iou = best_match
            matched_gt_indexes.add(gt_index)
            field_matches, field_total = _score_fields(pred_row=pred_row, gt_row=gt_row)
            field_accuracy = field_matches / float(max(field_total, 1))
            matched_rows.append(
                MatchedRow(
                    pred_index=pred_index,
                    gt_index=gt_index,
                    iou=round(iou, 6),
                    field_matches=field_matches,
                    field_total=field_total,
                    field_accuracy=round(field_accuracy, 6),
                    is_correct=field_accuracy >= row_field_accuracy_threshold,
                )
            )

        unmatched_ground_truth += max(len(ground_truth) - len(matched_gt_indexes), 0)

    correct_rows = sum(1 for row in matched_rows if row.is_correct)
    total_gt = len(gt_rows)
    return {
        "pred_rows": len(pred_rows),
        "gt_rows": total_gt,
        "matched_rows": len(matched_rows),
        "correct_rows": correct_rows,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_ground_truth": unmatched_ground_truth,
        "score": round(correct_rows / float(max(total_gt, 1)), 6),
        "mean_field_accuracy": _mean(row.field_accuracy for row in matched_rows),
        "mean_iou": _mean(row.iou for row in matched_rows),
        "iou_threshold": iou_threshold,
        "row_field_accuracy_threshold": row_field_accuracy_threshold,
        "matches": [
            {
                "pred_index": row.pred_index,
                "gt_index": row.gt_index,
                "iou": row.iou,
                "field_matches": row.field_matches,
                "field_total": row.field_total,
                "field_accuracy": row.field_accuracy,
                "is_correct": row.is_correct,
            }
            for row in matched_rows
        ],
    }


def read_ground_truth_rows(root: Path) -> list[dict[str, str]]:
    if root.is_file():
        return read_price_tag_csv(root)

    rows: list[dict[str, str]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path.name.lower() == "sample.csv":
            continue
        rows.extend(read_price_tag_csv(csv_path))
    if not rows:
        raise FileNotFoundError(f"No ground-truth CSV files found under {root}.")
    return rows


def read_price_tag_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = _normalize_row_keys(row)
            if not normalized.get("filename"):
                normalized["filename"] = _infer_video_filename_from_csv(path)
            rows.append({column: normalized.get(column, "") for column in CSV_COLUMNS})
        return rows


def _normalize_row_keys(row: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        canonical_key = HEADER_ALIASES.get(key, key)
        normalized[canonical_key] = "" if value is None else str(value).strip()
    return normalized


def _infer_video_filename_from_csv(path: Path) -> str:
    return f"{path.stem}.mp4"


def _group_by_video(rows: list[dict[str, str]]) -> dict[str, list[tuple[int, dict[str, str]]]]:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(rows):
        filename = Path(row.get("filename") or "").name
        if not filename:
            filename = "unknown"
        grouped.setdefault(filename, []).append((index, row))
    return grouped


def _row_bbox(row: dict[str, str]) -> BoundingBox | None:
    try:
        return BoundingBox(
            x_min=int(round(_parse_number(row.get("x_min")))),
            y_min=int(round(_parse_number(row.get("y_min")))),
            x_max=int(round(_parse_number(row.get("x_max")))),
            y_max=int(round(_parse_number(row.get("y_max")))),
        )
    except (ValueError, TypeError):
        return None


def _score_fields(*, pred_row: dict[str, str], gt_row: dict[str, str]) -> tuple[int, int]:
    matches = 0
    total = 0
    for field in CSV_COLUMNS:
        if field in IGNORED_FIELD_COLUMNS:
            continue
        gt_value = gt_row.get(field, "")
        if _is_empty(gt_value):
            continue
        total += 1
        if field_matches(field=field, pred_value=pred_row.get(field, ""), gt_value=gt_value):
            matches += 1
    return matches, total


def field_matches(*, field: str, pred_value: str | None, gt_value: str | None) -> bool:
    pred = _normalize_text(pred_value)
    gt = _normalize_text(gt_value)
    if _is_empty(gt):
        return _is_empty(pred)

    if field in DIGIT_ONLY_FIELDS:
        return _digits(pred) == _digits(gt)
    if field in PRICE_FIELDS:
        return _prices_equal(pred, gt)
    if field == "print_datetime":
        return _normalize_datetime(pred) == _normalize_datetime(gt)
    if field in TEXT_SIMILARITY_FIELDS:
        if not pred:
            return False
        return _similarity(pred, gt) >= 0.85
    return pred.casefold() == gt.casefold()


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _is_empty(value: str | None) -> bool:
    return _normalize_text(value) == ""


def _digits(value: str) -> str:
    return re.sub(r"\D+", "", value)


def _prices_equal(left: str, right: str) -> bool:
    left_number = _parse_optional_decimal(left)
    right_number = _parse_optional_decimal(right)
    if left_number is None or right_number is None:
        return left.casefold() == right.casefold()
    return abs(left_number - right_number) <= Decimal("0.01")


def _parse_number(value: str | None) -> float:
    parsed = _parse_optional_decimal(_normalize_text(value))
    if parsed is None:
        raise ValueError(f"Invalid numeric value: {value!r}")
    return float(parsed)


def _parse_optional_decimal(value: str | None) -> Decimal | None:
    text = _normalize_text(value)
    if not text or text.casefold() == "нет":
        return None
    text = text.replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", "."}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _normalize_datetime(value: str) -> str:
    match = re.search(
        r"(?P<date>\d{1,2}[.]\d{1,2}[.]\d{2,4})(?:\s+(?P<time>\d{1,2}:\d{2}))?",
        value,
    )
    if match is None:
        return _normalize_text(value).casefold()
    date = match.group("date")
    time = match.group("time") or ""
    return f"{date} {time}".strip()


def _similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]
    except ImportError:
        return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
    return float(fuzz.token_sort_ratio(left, right)) / 100.0


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 6)

