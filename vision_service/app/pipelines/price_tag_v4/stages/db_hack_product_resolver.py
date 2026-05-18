from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.pipelines.price_tag_v4.stages.catalog_builder import (
    bool_value,
    digits,
    load_db_hack_catalog,
    resolve_repo_path,
)
from app.schemas.detections import CropCandidate, DecodedSymbol


@dataclass(frozen=True, slots=True)
class DbHackResolverConfig:
    enabled: bool
    db_hack_path: Path
    repair_one_digit: bool


class DbHackProductResolverStage(BaseStage):
    name = "DbHackProductResolverStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "db_hack_path": str(config.db_hack_path),
            "crops_count": len(context.crop_candidates),
            "decoded_symbols_count": len(context.decoded_symbols),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled:
            context.artifacts["db_hack_product_resolver"] = {"enabled": False}
            return StageOutcome(output_summary=context.artifacts["db_hack_product_resolver"])

        warnings: list[str] = []
        try:
            catalog = load_db_hack_catalog(config.db_hack_path)
        except Exception as exc:  # noqa: BLE001
            context.artifacts["db_hack_product_resolver"] = {
                "enabled": True,
                "resolved_tracks": 0,
                "catalog_codes": 0,
            }
            return StageOutcome(
                output_summary=context.artifacts["db_hack_product_resolver"],
                warnings=[f"db_hack resolver could not load catalog: {exc}"],
            )

        crops_by_detection = {crop.detection_id: crop for crop in context.crop_candidates}
        track_candidates: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)

        for crop in context.crop_candidates:
            track_id = crop_track_id(crop)
            for code, confidence, source in barcode_candidates_from_crop(
                crop,
                repair_one_digit=config.repair_one_digit,
                valid_codes=catalog.code_to_name,
            ):
                name = catalog.name_for(code)
                if name:
                    track_candidates[track_id].append((code, name, confidence, source))

        for symbol in context.decoded_symbols:
            crop = crops_by_detection.get(symbol.detection_id)
            if crop is None:
                continue
            track_id = crop_track_id(crop)
            for code in barcode_candidates_from_symbol(symbol):
                name = catalog.name_for(code)
                if name:
                    track_candidates[track_id].append((code, name, 0.99, f"symbol:{symbol.decoder}"))

        resolved_tracks = 0
        updated_crops = 0
        notes: list[dict[str, Any]] = []
        for track_id, candidates in track_candidates.items():
            if not candidates:
                continue
            code, name, confidence, source = max(candidates, key=lambda item: item[2])
            track_crops = [crop for crop in context.crop_candidates if crop_track_id(crop) == track_id]
            if not track_crops:
                continue
            best_crop = max(track_crops, key=lambda crop: crop.quality.score)
            merge_catalog_identity(
                best_crop,
                barcode=code,
                product_name=name,
                confidence=confidence,
                source=source,
            )
            resolved_tracks += 1
            updated_crops += 1
            notes.append(
                {
                    "track_id": track_id,
                    "barcode": code,
                    "product_name": name,
                    "confidence": round(confidence, 4),
                    "source": source,
                }
            )

        context.artifacts["db_hack_product_resolver"] = {
            "enabled": True,
            "catalog_codes": len(catalog.code_to_name),
            "resolved_tracks": resolved_tracks,
            "updated_crops": updated_crops,
            "notes": notes[:50],
            "mode_note": "Uses only visual barcode/QR/OCR digit evidence plus db_hack barcode->fullname.",
        }
        return StageOutcome(
            output_summary=context.artifacts["db_hack_product_resolver"],
            warnings=warnings,
        )


def config_from_context(context: PipelineContext) -> DbHackResolverConfig:
    raw = context.config.get("db_hack_product_resolver", {})
    if not isinstance(raw, dict):
        raw = {}
    return DbHackResolverConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        db_hack_path=resolve_repo_path(str(raw.get("db_hack_path") or "data/db_hack.csv")),
        repair_one_digit=bool_value(raw.get("repair_one_digit"), default=True),
    )


def barcode_candidates_from_symbol(symbol: DecodedSymbol) -> list[str]:
    parsed = parse_qr_payload(symbol.payload)
    values = [parsed.get("barcode", ""), parsed.get("qr_code_barcode", ""), symbol.payload]
    return [code for code in (normalize_barcode(value) for value in values) if code]


def barcode_candidates_from_crop(
    crop: CropCandidate,
    *,
    repair_one_digit: bool,
    valid_codes: dict[str, str],
) -> list[tuple[str, float, str]]:
    payload = crop.attributes.get("ocr")
    if not isinstance(payload, dict):
        return []
    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    text_values = [
        str(payload.get("text", "")),
        *(str(value) for value in field_values.values()),
    ]
    candidates: list[tuple[str, float, str]] = []
    for value in text_values:
        for raw_digits in digit_windows(value):
            code = normalize_barcode(raw_digits)
            if code and code in valid_codes:
                candidates.append((code, 0.96, "ocr_exact_ean13"))
                continue
            if len(raw_digits) == 12:
                candidate = raw_digits + ean13_checksum(raw_digits)
                if candidate in valid_codes:
                    candidates.append((candidate, 0.84, "ocr_12_plus_checksum"))
                    continue
            if repair_one_digit and len(raw_digits) == 13:
                fixed = repair_one_digit_ean13(raw_digits, valid_codes)
                if fixed:
                    candidates.append((fixed, 0.78, "ocr_one_digit_repair"))
    return candidates


def merge_catalog_identity(
    crop: CropCandidate,
    *,
    barcode: str,
    product_name: str,
    confidence: float,
    source: str,
) -> None:
    existing = crop.attributes.get("ocr")
    if not isinstance(existing, dict):
        existing = {}
    fields = dict(existing.get("fields", {})) if isinstance(existing.get("fields"), dict) else {}
    fields["barcode"] = barcode
    fields["qr_code_barcode"] = barcode
    fields["product_name"] = product_name
    crop.attributes["ocr"] = {
        **existing,
        "fields": fields,
        "confidence": max(float(existing.get("confidence", 0.0) or 0.0), confidence),
        "engine": source,
        "catalog_resolver": "db_hack_product_resolver",
    }


def crop_track_id(crop: CropCandidate) -> str:
    value = crop.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return crop.detection_id


def digit_windows(value: str) -> list[str]:
    compact = digits(value)
    windows: list[str] = []
    if len(compact) in {12, 13, 14}:
        windows.append(compact)
    if len(compact) > 13:
        windows.extend(compact[index : index + 13] for index in range(0, len(compact) - 12))
        windows.extend(compact[index : index + 12] for index in range(0, len(compact) - 11))
    return list(dict.fromkeys(windows))


def normalize_barcode(value: str) -> str:
    code = digits(value)
    if len(code) == 14 and code.startswith("0"):
        code = code[1:]
    if len(code) == 13 and ean13_ok(code):
        return code
    return ""


def repair_one_digit_ean13(value: str, valid_codes: dict[str, str]) -> str:
    code = digits(value)
    if len(code) != 13:
        return ""
    for index in range(13):
        for digit in "0123456789":
            candidate = code[:index] + digit + code[index + 1 :]
            if candidate != code and ean13_ok(candidate) and candidate in valid_codes:
                return candidate
    return ""


def ean13_checksum(first_12: str) -> str:
    checksum = sum((1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(first_12))
    return str((10 - checksum % 10) % 10)


def ean13_ok(value: str) -> bool:
    code = digits(value)
    return bool(re.fullmatch(r"\d{13}", code)) and ean13_checksum(code[:12]) == code[12]
