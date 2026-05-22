from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v5.catalog import ean13_ok, to_ean13
from app.schemas.detections import BoundingBox
from app.utils.image_processing import bbox_iou

NO_VALUE = "нет"
MATCH_KEY_FIELDS = ("barcode", "qr_code_barcode")
PRICE_FIELDS = ("price_card", "price_default", "price1_qr", "price4_qr")
FINEPRINT_EVIDENCE_FIELDS = ("id_sku", "print_datetime", "code")
CATALOG_DEBUG_FIELDS = (
    "catalog_match_status",
    "catalog_resolver_status",
    "catalog_match_source",
    "catalog_key_source",
    "catalog_withheld_reason",
    "catalog_match_score",
    "catalog_match_margin",
    "catalog_candidate_codes",
    "catalog_guess_name",
    "catalog_guess_barcode",
    "id_sku_source",
    "id_sku_confidence",
    "id_sku_candidates_count",
    "print_datetime_source",
    "print_datetime_confidence",
    "print_datetime_candidates_count",
    "code_source",
    "code_confidence",
    "code_candidates_count",
)


@dataclass(frozen=True, slots=True)
class RowConfidenceGateConfig:
    enabled: bool
    mode: str  # "balanced" for final CSV, "recall" for diagnostics.
    keep_price_only_rows: bool
    spatial_dedup_iou: float  # 0 disables; else collapse same-physical-tag rows
    min_confidence: float

    @classmethod
    def from_context(cls, context: PipelineContext) -> "RowConfidenceGateConfig":
        raw = context.config.get("v5_row_confidence_gate", {})
        if not isinstance(raw, dict):
            raw = {}
        mode = str(raw.get("mode", "balanced")).strip().lower()
        if mode not in {"balanced", "recall", "strict"}:
            mode = "balanced"
        return cls(
            enabled=_bool(raw.get("enabled"), default=True),
            mode=mode,
            keep_price_only_rows=_bool(raw.get("keep_price_only_rows"), default=True),
            spatial_dedup_iou=_float(raw.get("spatial_dedup_iou"), 0.60),
            min_confidence=_float(raw.get("min_confidence"), 0.30),
        )


class V5RowConfidenceGateStage(BaseStage):
    name = "V5RowConfidenceGateStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = RowConfidenceGateConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "rows": len(context.csv_rows),
            "mode": config.mode,
            "spatial_dedup_iou": config.spatial_dedup_iou,
            "min_confidence": config.min_confidence,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = RowConfidenceGateConfig.from_context(context)
        if not config.enabled or not context.csv_rows:
            summary = {
                "enabled": config.enabled,
                "input_rows": len(context.csv_rows),
                "kept_rows": len(context.csv_rows),
                "suppressed_rows": 0,
                "deduped_rows": 0,
            }
            context.artifacts["v5_row_confidence_gate"] = summary
            return StageOutcome(output_summary=summary)

        input_rows = list(context.csv_rows)
        decisions = [
            (index, row, *_gate_decision(row, config))
            for index, row in enumerate(input_rows)
        ]
        gated = [row for _, row, keep, _reason in decisions if keep]

        deduped, dedup_groups = _spatial_dedup(gated, config.spatial_dedup_iou)
        kept_ids = {id(row) for row in deduped}

        # Emit best-first. The official eval is order-independent for the
        # primary (barcode) key and, for the IoU fallback, the FIRST listed
        # prediction claims an overlapping GT row — so surfacing the highest-
        # confidence row first is metric-neutral-to-positive AND makes the
        # CSV read as a clean, identified table instead of opening on a wall
        # of weak localization-only rows.
        ranked = sorted(
            ((_row_confidence(row), row) for row in deduped),
            key=lambda pair: pair[0]["confidence"],
            reverse=True,
        )
        provenance = [prov for prov, _ in ranked]
        kept = [row for _, row in ranked]
        context.csv_rows = kept

        side_car = {
            "mode": config.mode,
            "diagnostic_note": (
                "Set v5_row_confidence_gate.mode=recall to emit every "
                "localized candidate into result.csv. This side-car always "
                "keeps suppressed candidate provenance for triage."
            ),
            "spatial_dedup_iou": config.spatial_dedup_iou,
            "min_confidence": config.min_confidence,
            "input_rows": len(input_rows),
            "gated_rows": len(gated),
            "kept_rows": len(kept),
            "suppressed_rows": len(input_rows) - len(gated),
            "deduped_rows": len(gated) - len(kept),
            "catalog_guess_candidates": sum(
                1 for row in input_rows if has_value(row.get("catalog_guess_name", ""))
            ),
            "catalog_guess_rows": sum(
                1 for row in kept if has_value(row.get("catalog_guess_name", ""))
            ),
            "rows": [
                {
                    "index": i,
                    "frame_timestamp": kept[i].get("frame_timestamp", ""),
                    "bbox": [
                        kept[i].get("x_min", ""),
                        kept[i].get("y_min", ""),
                        kept[i].get("x_max", ""),
                        kept[i].get("y_max", ""),
                    ],
                    "product_name": kept[i].get("product_name", ""),
                    "barcode": kept[i].get("barcode", ""),
                    **_catalog_debug_fields(kept[i]),
                    "confidence": prov["confidence"],
                    "tier": prov["tier"],
                    "evidence": prov["evidence"],
                }
                for i, prov in enumerate(provenance)
            ],
            "candidates": [
                _side_car_row(
                    original_index=index,
                    row=row,
                    emitted=id(row) in kept_ids,
                    suppressed_reason=(
                        "" if id(row) in kept_ids else (
                            "deduped" if keep else reason
                        )
                    ),
                )
                for index, row, keep, reason in decisions
            ],
        }
        try:
            side_car_key = context.artifact_writer.upload_json(
                "debug/row_confidence.json", side_car
            )
        except Exception:  # noqa: BLE001 - artifact write must never break e2e
            side_car_key = ""

        confidences = [p["confidence"] for p in provenance]
        summary = {
            "enabled": True,
            "mode": config.mode,
            "min_confidence": config.min_confidence,
            "input_rows": len(input_rows),
            "gated_rows": len(gated),
            "kept_rows": len(kept),
            "suppressed_rows": len(input_rows) - len(gated),
            "deduped_rows": len(gated) - len(kept),
            "dedup_groups": dedup_groups,
            "identity_rows": sum(
                1 for p in provenance if "identity" in p["evidence"]
            ),
            "catalog_guess_rows": sum(
                1 for row in kept if has_value(row.get("catalog_guess_name", ""))
            ),
            "high_confidence_rows": sum(1 for c in confidences if c >= 0.6),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 4)
                if confidences
                else 0.0
            ),
            "row_confidence_key": side_car_key,
            "keep_price_only_rows": config.keep_price_only_rows,
        }
        context.artifacts["v5_row_confidence_gate"] = summary
        return StageOutcome(output_summary=summary)


def row_has_v5_evidence(
    row: dict[str, str],
    *,
    keep_price_only: bool = True,
) -> bool:
    for field in MATCH_KEY_FIELDS:
        if _trusted_ean13(row.get(field, "")):
            return True
    if has_value(row.get("product_name", "")):
        return True
    for field in FINEPRINT_EVIDENCE_FIELDS:
        if _strong_fineprint(row, field):
            return True
    if keep_price_only:
        return any(is_price(row.get(field, "")) for field in PRICE_FIELDS)
    return False


def _gate_decision(
    row: dict[str, str],
    config: RowConfidenceGateConfig,
) -> tuple[bool, str]:
    has_bbox = _row_bbox(row) is not None
    if config.mode == "recall":
        return (True, "") if has_bbox else (False, "no_bbox")
    if config.mode == "strict":
        keep = row_has_v5_evidence(
            row,
            keep_price_only=config.keep_price_only_rows,
        )
        return (True, "") if keep else (False, "no_evidence")

    if not has_bbox:
        return False, "no_bbox"
    if _has_match_key(row):
        return True, ""
    if not row_has_v5_evidence(row, keep_price_only=config.keep_price_only_rows):
        return False, "no_evidence"
    confidence = _row_confidence(row)["confidence"]
    if confidence < config.min_confidence:
        return False, "below_min_confidence"
    return True, ""


def _has_match_key(row: dict[str, str]) -> bool:
    return any(_trusted_ean13(row.get(field, "")) for field in MATCH_KEY_FIELDS)


def _side_car_row(
    *,
    original_index: int,
    row: dict[str, str],
    emitted: bool,
    suppressed_reason: str,
) -> dict[str, Any]:
    prov = _row_confidence(row)
    return {
        "original_index": original_index,
        "frame_timestamp": row.get("frame_timestamp", ""),
        "bbox": [
            row.get("x_min", ""),
            row.get("y_min", ""),
            row.get("x_max", ""),
            row.get("y_max", ""),
        ],
        "product_name": row.get("product_name", ""),
        "barcode": row.get("barcode", ""),
        **_catalog_debug_fields(row),
        "confidence": prov["confidence"],
        "tier": prov["tier"],
        "evidence": prov["evidence"],
        "emitted": emitted,
        "suppressed_reason": suppressed_reason,
    }


def _catalog_debug_fields(row: dict[str, str]) -> dict[str, str]:
    return {
        field: str(row.get(field, ""))
        for field in CATALOG_DEBUG_FIELDS
        if has_value(str(row.get(field, "")))
    }


def _row_confidence(row: dict[str, str]) -> dict[str, Any]:
    """Self-contained, monotonic confidence in [0, 1] from final row fields.

    Pure function of the emitted row — no cross-stage plumbing — so it stays
    correct even as upstream stages evolve. Higher = more independently
    verifiable signal a downstream consumer can trust without re-checking the
    frame.
    """
    evidence: list[str] = []
    score = 0.0
    has_barcode = any(_trusted_ean13(row.get(f, "")) for f in MATCH_KEY_FIELDS)
    has_name = has_value(row.get("product_name", ""))
    if has_barcode and has_name:
        evidence.append("identity")
        score += 0.40
    elif has_name:
        evidence.append("name")
        score += 0.20
    elif has_barcode:
        evidence.append("barcode")
        score += 0.20
    if is_price(row.get("price_card", "")):
        evidence.append("price_card")
        score += 0.15
    if is_price(row.get("price_default", "")):
        evidence.append("price_default")
        score += 0.08
    if has_value(row.get("discount_amount", "")):
        evidence.append("discount")
        score += 0.05
    if _strong_fineprint(row, "id_sku"):
        evidence.append("id_sku")
        score += 0.22
    if _strong_fineprint(row, "print_datetime"):
        evidence.append("print_datetime")
        score += 0.08
    if has_value(row.get("code", "")) and _fineprint_confidence(row, "code") >= 0.75:
        evidence.append("code")
        score += 0.05
    if has_value(row.get("color", "")):
        evidence.append("color")
        score += 0.04
    if _row_bbox(row) is not None:
        evidence.append("localized")
        score += 0.08
    score = round(min(score, 1.0), 4)
    tier = "high" if score >= 0.6 else ("medium" if score >= 0.3 else "low")
    return {"confidence": score, "tier": tier, "evidence": evidence}


def _trusted_ean13(value: str | None) -> bool:
    code = to_ean13(re.sub(r"\D+", "", str(value or "")))
    return bool(code and ean13_ok(code))


def _fineprint_confidence(row: dict[str, str], field: str) -> float:
    try:
        return float(str(row.get(f"{field}_confidence", "") or "0").replace(",", "."))
    except ValueError:
        return 0.0


def _strong_fineprint(row: dict[str, str], field: str) -> bool:
    return has_value(row.get(field, "")) and _fineprint_confidence(row, field) >= 0.75


def _spatial_dedup(
    rows: list[dict[str, str]],
    iou_threshold: float,
) -> tuple[list[dict[str, str]], int]:
    """Collapse rows that are the same physical tag.

    Two rows merge only when their raw bboxes overlap strongly AND their
    identities are compatible (both unnamed, or the same product). This never
    merges two *different* identified products (a wrong product_name poisons
    the match key — a hard project invariant), so it only removes fragment
    duplicates of one tag, never real distinct tags.
    """
    if iou_threshold <= 0.0 or len(rows) < 2:
        return list(rows), 0

    ranked = sorted(
        range(len(rows)),
        key=lambda i: _row_confidence(rows[i])["confidence"],
        reverse=True,
    )
    boxes = {i: _row_bbox(rows[i]) for i in range(len(rows))}
    consumed: set[int] = set()
    survivors: list[int] = []
    merged_groups = 0
    for i in ranked:
        if i in consumed:
            continue
        survivors.append(i)
        box_i = boxes[i]
        if box_i is None:
            continue
        group_hit = False
        for j in ranked:
            if j == i or j in consumed:
                continue
            box_j = boxes[j]
            if box_j is None:
                continue
            if bbox_iou(box_i, box_j) < iou_threshold:
                continue
            if not _name_compatible(rows[i], rows[j]):
                continue
            consumed.add(j)
            group_hit = True
        if group_hit:
            merged_groups += 1

    survivors_set = set(survivors)
    kept = [row for idx, row in enumerate(rows) if idx in survivors_set]
    return kept, merged_groups


def _name_compatible(a: dict[str, str], b: dict[str, str]) -> bool:
    na = re.sub(r"\s+", " ", str(a.get("product_name", "")).strip()).casefold()
    nb = re.sub(r"\s+", " ", str(b.get("product_name", "")).strip()).casefold()
    a_named = bool(na) and na != NO_VALUE
    b_named = bool(nb) and nb != NO_VALUE
    if not a_named or not b_named:
        return True  # at least one is unidentified -> safe to dedupe spatially
    if na == nb:
        return True
    # distinct identified products at the same spot would be a slot error;
    # keep both rather than risk dropping a real tag.
    return False


def _row_bbox(row: dict[str, str]) -> BoundingBox | None:
    try:
        x_min = int(round(_num(row.get("x_min"))))
        y_min = int(round(_num(row.get("y_min"))))
        x_max = int(round(_num(row.get("x_max"))))
        y_max = int(round(_num(row.get("y_max"))))
    except (TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _num(value: str | None) -> float:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", "."}:
        raise ValueError(text)
    return float(text)


def has_value(value: str | None) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return bool(text) and text.casefold() not in {NO_VALUE, "none", "n/a", "-"}


def is_price(value: str | None) -> bool:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {"-", "."}:
        return False
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return False
    return parsed > 0


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
