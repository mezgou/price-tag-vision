from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import CropCandidate


@dataclass(slots=True)
class _ConsensusConfig:
    enabled: bool


class CharConsensusStage(BaseStage):
    name = "CharConsensusStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "decoded_symbols_count": len(context.decoded_symbols),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        if not config.enabled:
            return StageOutcome(output_summary={"enabled": False})

        by_track: dict[str, list[CropCandidate]] = defaultdict(list)
        track_by_detection: dict[str, str] = {}
        for crop in context.crop_candidates:
            track_id = _track_id(crop)
            by_track[track_id].append(crop)
            track_by_detection[crop.detection_id] = track_id

        corrected_fields = 0
        tracks_with_consensus = 0
        for track_id, crops in by_track.items():
            fields_by_name: dict[str, list[str]] = defaultdict(list)
            for crop in crops:
                for field, value in _ocr_fields(crop).items():
                    fields_by_name[field].append(value)
            for symbol in context.decoded_symbols:
                if _symbol_track_id(
                    symbol.attributes,
                    symbol.detection_id,
                    track_by_detection,
                ) == track_id:
                    if symbol.symbol_type == "barcode":
                        fields_by_name["barcode"].append(symbol.payload)

            consensus: dict[str, str] = {}
            barcode = _consensus_ean13(
                fields_by_name.get("barcode", []) + fields_by_name.get("qr_code_barcode", [])
            )
            if barcode:
                consensus["barcode"] = barcode
                consensus["qr_code_barcode"] = barcode
            sku = _consensus_fixed_digits(fields_by_name.get("id_sku", []), length=12)
            if sku:
                consensus["id_sku"] = sku
            for field in ("price_default", "price_card", "price1_qr", "price2_qr", "price4_qr"):
                price = _consensus_price(fields_by_name.get(field, []))
                if price:
                    consensus[field] = price
            date = _consensus_date(fields_by_name.get("print_datetime", []))
            if date:
                consensus["print_datetime"] = date

            if not consensus:
                continue
            tracks_with_consensus += 1
            best_crop = max(crops, key=lambda item: item.quality.score)
            _merge_fields(best_crop, consensus)
            corrected_fields += len(consensus)

        summary = {
            "enabled": True,
            "tracks_with_consensus": tracks_with_consensus,
            "corrected_fields": corrected_fields,
        }
        context.artifacts["char_consensus"] = summary
        return StageOutcome(output_summary=summary)


def _track_id(crop: CropCandidate) -> str:
    value = crop.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return crop.detection_id


def _symbol_track_id(
    attributes: dict[str, Any],
    detection_id: str,
    track_by_detection: dict[str, str],
) -> str:
    value = attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return track_by_detection.get(detection_id, detection_id)


def _ocr_fields(crop: CropCandidate) -> dict[str, str]:
    payload = crop.attributes.get("ocr")
    if not isinstance(payload, dict):
        return {}
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {str(key): str(value) for key, value in fields.items() if str(value).strip()}


def _merge_fields(crop: CropCandidate, fields: dict[str, str]) -> None:
    ocr = crop.attributes.get("ocr")
    if not isinstance(ocr, dict):
        ocr = {}
    merged = dict(ocr.get("fields", {})) if isinstance(ocr.get("fields"), dict) else {}
    merged.update(fields)
    crop.attributes["ocr"] = {
        **ocr,
        "fields": merged,
        "engine": ocr.get("engine", "char_consensus"),
        "confidence": max(float(ocr.get("confidence", 0.0) or 0.0), 0.86),
    }


def _consensus_ean13(values: list[str]) -> str:
    candidates: list[str] = []
    for value in values:
        digits = re.sub(r"\D+", "", value)
        if len(digits) == 12:
            candidates.append(digits + _ean13_checksum(digits))
        elif len(digits) == 13 and _ean13_ok(digits):
            candidates.append(digits)
        elif len(digits) == 13:
            fixed = _fix_one_ean13_digit(digits)
            if fixed:
                candidates.append(fixed)
    if not candidates:
        return ""
    voted = _vote_fixed(candidates, length=13)
    if _ean13_ok(voted):
        return voted
    common, count = Counter(candidates).most_common(1)[0]
    return common if count >= 1 and _ean13_ok(common) else ""


def _fix_one_ean13_digit(value: str) -> str:
    for index in range(13):
        for digit in "0123456789":
            candidate = value[:index] + digit + value[index + 1 :]
            if candidate != value and _ean13_ok(candidate):
                return candidate
    return ""


def _ean13_checksum(first_12: str) -> str:
    checksum = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(first_12))
    return str((10 - checksum % 10) % 10)


def _ean13_ok(value: str) -> bool:
    return bool(re.fullmatch(r"\d{13}", value or "")) and _ean13_checksum(value[:12]) == value[12]


def _consensus_fixed_digits(values: list[str], *, length: int) -> str:
    candidates = [re.sub(r"\D+", "", value) for value in values]
    candidates = [value for value in candidates if len(value) == length]
    if not candidates:
        return ""
    return _vote_fixed(candidates, length=length)


def _vote_fixed(values: list[str], *, length: int) -> str:
    chars = []
    for index in range(length):
        counter = Counter(value[index] for value in values if len(value) == length)
        chars.append(counter.most_common(1)[0][0] if counter else "")
    return "".join(chars)


def _consensus_price(values: list[str]) -> str:
    normalized = [_normalize_price(value) for value in values]
    normalized = [value for value in normalized if value]
    if not normalized:
        return ""
    return Counter(normalized).most_common(1)[0][0]


def _normalize_price(value: str) -> str:
    text = str(value).replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return ""
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return ""
    if parsed < Decimal("0") or parsed > Decimal("100000"):
        return ""
    return f"{parsed:.2f}"


def _consensus_date(values: list[str]) -> str:
    matches = []
    for value in values:
        match = re.search(
            r"\d{1,2}[.]\d{1,2}[.]\d{2,4}(?:\s+\d{1,2}:\d{2})?",
            value,
        )
        if match:
            matches.append(match.group(0))
    if not matches:
        return ""
    return Counter(matches).most_common(1)[0][0]


def _config(context: PipelineContext) -> _ConsensusConfig:
    raw = context.config.get("char_consensus", {})
    if not isinstance(raw, dict):
        raw = {}
    return _ConsensusConfig(enabled=_bool(raw.get("enabled"), default=True))


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
