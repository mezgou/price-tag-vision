from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.pipelines.price_tag_v4.stages.catalog_builder import (
    load_db_hack_catalog,
    resolve_repo_path,
)
from app.pipelines.price_tag_v5.catalog import ean13_ok, to_ean13
from app.schemas.detections import DecodedSymbol

BARCODE_FIELDS = {"barcode", "qr_code_barcode"}
QR_REVERSE_FIELD_MAP = {
    "qr_code_barcode": "barcode",
    "price1_qr": "p1",
    "price2_qr": "p2",
    "price3_qr": "p3",
    "price4_qr": "p4",
    "wholesale_level_1_count": "wl1c",
    "wholesale_level_1_price": "wl1p",
    "wholesale_level_2_count": "wl2c",
    "wholesale_level_2_price": "wl2p",
    "action_price_qr": "ap",
    "action_code_qr": "ac",
}


@dataclass(frozen=True, slots=True)
class V5DecodedSymbolGateConfig:
    enabled: bool
    db_hack_path: Path
    require_ean13_checksum: bool
    require_catalog_code_for_match_key: bool
    keep_rejected_symbols_artifact: bool

    @classmethod
    def from_context(cls, context: PipelineContext) -> "V5DecodedSymbolGateConfig":
        raw = context.config.get("v5_decoded_symbol_gate", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=_bool(raw.get("enabled"), default=True),
            db_hack_path=resolve_repo_path(str(raw.get("db_hack_path") or "data/db_hack.csv")),
            require_ean13_checksum=_bool(raw.get("require_ean13_checksum"), default=True),
            require_catalog_code_for_match_key=_bool(
                raw.get("require_catalog_code_for_match_key"),
                default=True,
            ),
            keep_rejected_symbols_artifact=_bool(
                raw.get("keep_rejected_symbols_artifact"),
                default=True,
            ),
        )


class V5DecodedSymbolGateStage(BaseStage):
    name = "V5DecodedSymbolGateStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = V5DecodedSymbolGateConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "decoded_symbols": len(context.decoded_symbols),
            "require_ean13_checksum": config.require_ean13_checksum,
            "require_catalog_code_for_match_key": config.require_catalog_code_for_match_key,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = V5DecodedSymbolGateConfig.from_context(context)
        if not config.enabled:
            return StageOutcome(
                output_summary={
                    "enabled": False,
                    "input_symbols": len(context.decoded_symbols),
                    "kept_symbols": len(context.decoded_symbols),
                }
            )

        valid_codes: dict[str, str] = {}
        warnings: list[str] = []
        if config.require_catalog_code_for_match_key:
            try:
                valid_codes = _normalized_catalog_codes(
                    load_db_hack_catalog(config.db_hack_path).code_to_name
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"v5 decoded-symbol gate could not load db_hack: {exc}")

        det2track = {
            detection.detection_id: str(
                detection.attributes.get("track_id") or detection.detection_id
            )
            for detection in context.detections
        }
        kept: list[DecodedSymbol] = []
        rejected: list[dict[str, Any]] = []
        evidence_by_track: dict[str, list[dict[str, Any]]] = {}
        for symbol in context.decoded_symbols:
            sanitized, evidence, reason = _sanitize_symbol(
                symbol,
                config=config,
                valid_codes=valid_codes,
            )
            if sanitized is None:
                rejected.append(_symbol_debug(symbol, reason=reason))
                continue
            kept.append(sanitized)
            track_id = det2track.get(symbol.detection_id) or str(
                symbol.attributes.get("track_id") or symbol.detection_id
            )
            for item in evidence:
                evidence_by_track.setdefault(track_id, []).append(
                    {
                        **item,
                        "track_id": track_id,
                        "symbol_id": symbol.symbol_id,
                        "crop_id": symbol.crop_id,
                        "detection_id": symbol.detection_id,
                        "decoder": symbol.decoder,
                        "variant": symbol.variant,
                        "confidence": symbol.confidence,
                    }
                )

        context.decoded_symbols = kept
        context.artifacts["v5_code_evidence_by_track"] = evidence_by_track
        rejected_key = ""
        if config.keep_rejected_symbols_artifact and rejected:
            try:
                rejected_key = context.artifact_writer.upload_json(
                    "debug/v5_rejected_decoded_symbols.json",
                    {"symbols": rejected},
                )
            except Exception:  # noqa: BLE001 - debug artifact only
                rejected_key = ""
        summary = {
            "enabled": True,
            "input_symbols": len(kept) + len(rejected),
            "kept_symbols": len(kept),
            "rejected_symbols": len(rejected),
            "tracks_with_code_evidence": len(evidence_by_track),
            "rejected_symbols_key": rejected_key,
        }
        context.artifacts["v5_decoded_symbol_gate"] = summary
        return StageOutcome(output_summary=summary, warnings=warnings)


def _sanitize_symbol(
    symbol: DecodedSymbol,
    *,
    config: V5DecodedSymbolGateConfig,
    valid_codes: dict[str, str],
) -> tuple[DecodedSymbol | None, list[dict[str, Any]], str]:
    parsed = parse_qr_payload(symbol.payload)
    evidence: list[dict[str, Any]] = []
    if parsed:
        sanitized_fields: dict[str, str] = {}
        for field, value in parsed.items():
            if field in BARCODE_FIELDS:
                code, reason = _trusted_code(
                    value,
                    config=config,
                    valid_codes=valid_codes,
                )
                if not code:
                    continue
                sanitized_fields[field] = code
                evidence.append(
                    {
                        "code": code,
                        "field": field,
                        "source": f"symbol:{symbol.decoder}",
                        "catalog_product_name": valid_codes.get(code, ""),
                    }
                )
            else:
                sanitized_fields[field] = str(value)
        if not sanitized_fields:
            return None, [], "qr_payload_no_safe_fields"
        return (
            symbol.model_copy(
                update={
                    "payload": _encode_qr_fields(sanitized_fields),
                    "attributes": {
                        **symbol.attributes,
                        "v5_symbol_gate": "kept",
                        "safe_barcode_fields": [
                            field for field in sanitized_fields if field in BARCODE_FIELDS
                        ],
                    },
                }
            ),
            evidence,
            "",
        )

    code, reason = _trusted_code(
        symbol.payload,
        config=config,
        valid_codes=valid_codes,
    )
    if not code:
        return None, [], reason
    evidence.append(
        {
            "code": code,
            "field": "barcode" if symbol.symbol_type == "barcode" else "qr_code_barcode",
            "source": f"symbol:{symbol.decoder}",
            "catalog_product_name": valid_codes.get(code, ""),
        }
    )
    return (
        symbol.model_copy(
            update={
                "payload": code,
                "symbol_type": "barcode" if symbol.symbol_type != "qr" else symbol.symbol_type,
                "attributes": {
                    **symbol.attributes,
                    "v5_symbol_gate": "kept",
                    "ean13_valid": True,
                    "catalog_code_exists": bool(valid_codes.get(code)),
                },
            }
        ),
        evidence,
        "",
    )


def _trusted_code(
    value: str,
    *,
    config: V5DecodedSymbolGateConfig,
    valid_codes: dict[str, str],
) -> tuple[str, str]:
    raw_digits = re.sub(r"\D+", "", str(value or ""))
    code = to_ean13(raw_digits)
    if not code:
        return "", "no_usable_barcode_digits"
    if config.require_ean13_checksum and not ean13_ok(code):
        return "", "bad_ean13_checksum"
    if config.require_catalog_code_for_match_key and code not in valid_codes:
        return "", "barcode_not_in_catalog"
    return code, ""


def _encode_qr_fields(fields: dict[str, str]) -> str:
    return ";".join(
        f"{QR_REVERSE_FIELD_MAP.get(key, key)}={value}"
        for key, value in sorted(fields.items())
    )


def _normalized_catalog_codes(code_to_name: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_code, name in code_to_name.items():
        code = re.sub(r"\D+", "", str(raw_code or ""))
        if code:
            out.setdefault(code, name)
        normalized = to_ean13(code)
        if normalized and ean13_ok(normalized):
            out.setdefault(normalized, name)
    return out


def _symbol_debug(symbol: DecodedSymbol, *, reason: str) -> dict[str, Any]:
    return {
        "symbol_id": symbol.symbol_id,
        "crop_id": symbol.crop_id,
        "detection_id": symbol.detection_id,
        "symbol_type": symbol.symbol_type,
        "decoder": symbol.decoder,
        "variant": symbol.variant,
        "payload": symbol.payload,
        "confidence": symbol.confidence,
        "reason": reason,
    }


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default
