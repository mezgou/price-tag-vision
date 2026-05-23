from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import (
    _build_frame_lookup,
    _duration_ms,
    _load_crop_image,
)
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.pipelines.price_tag_v3.stages.barcode_bars_decode import (
    _barcode_band,
    _barcode_variants,
    _decode_pyzbar,
    _decode_signal,
    _decode_zxingcpp,
    _ean13_ok,
)
from app.pipelines.price_tag_v5.catalog import ean13_ok as _v5_ean13_ok
from app.pipelines.price_tag_v5.catalog import to_ean13
from app.schemas.detections import DecodeAttempt, DecodedSymbol


@dataclass(frozen=True, slots=True)
class V5BarcodeBarsCropConfig:
    enabled: bool
    max_crops: int
    min_crop_quality_score: float
    max_crops_per_track: int
    skip_if_code_evidence_present: bool
    min_confidence: float
    stop_after_first_success_per_track: bool

    @classmethod
    def from_context(cls, context: PipelineContext) -> "V5BarcodeBarsCropConfig":
        raw = context.config.get("v5_barcode_bars_decode", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=_bool(raw.get("enabled"), default=True),
            max_crops=_positive_int(raw.get("max_crops"), 300),
            min_crop_quality_score=_float(raw.get("min_crop_quality_score"), 0.05),
            max_crops_per_track=_non_negative_int(raw.get("max_crops_per_track"), 0),
            skip_if_code_evidence_present=_bool(
                raw.get("skip_if_code_evidence_present"),
                default=True,
            ),
            min_confidence=_float(raw.get("min_confidence"), 0.74),
            stop_after_first_success_per_track=_bool(
                raw.get("stop_after_first_success_per_track"),
                default=True,
            ),
        )


class V5BarcodeBarsCropDecodeStage(BaseStage):
    name = "V5BarcodeBarsCropDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = V5BarcodeBarsCropConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops": config.max_crops,
            "min_crop_quality_score": config.min_crop_quality_score,
            "max_crops_per_track": config.max_crops_per_track,
            "skip_if_code_evidence_present": config.skip_if_code_evidence_present,
            "min_confidence": config.min_confidence,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = V5BarcodeBarsCropConfig.from_context(context)
        if not config.enabled or not context.crop_candidates:
            return StageOutcome(
                output_summary={
                    "enabled": config.enabled,
                    "crops_processed": 0,
                    "decode_attempts": 0,
                    "hits": 0,
                }
            )

        frame_lookup = _build_frame_lookup(context.sampled_frames)
        quality_eligible = [
            crop
            for crop in sorted(
                context.crop_candidates,
                key=lambda item: item.quality.score,
                reverse=True,
            )
            if crop.quality.score >= config.min_crop_quality_score
        ]
        coded_tracks = (
            _ean_coded_tracks(context.decoded_symbols)
            if config.skip_if_code_evidence_present
            else set()
        )
        eligible = _cap_crops_per_track(
            quality_eligible,
            max_total=config.max_crops,
            max_per_track=config.max_crops_per_track,
            skip_track_ids=coded_tracks,
        )
        skipped_by_track_cap = max(len(quality_eligible) - len(eligible), 0)
        skipped_by_code_evidence = sum(
            1 for crop in quality_eligible if _track_id(crop) in coded_tracks
        )
        seen = {
            (symbol.detection_id, re.sub(r"\D+", "", symbol.payload))
            for symbol in context.decoded_symbols
        }
        tracks_with_hit: set[str] = set()
        attempts: list[DecodeAttempt] = []
        symbols: list[DecodedSymbol] = []
        warnings: list[str] = []
        crops_processed = 0

        for crop in eligible:
            track_id = _track_id(crop)
            if config.stop_after_first_success_per_track and track_id in tracks_with_hit:
                continue
            image = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
            if image is None:
                continue
            crops_processed += 1
            hit_value = ""
            hit_decoder = ""
            hit_variant = ""
            hit_confidence = 0.0
            for variant_name, variant in _barcode_variants(_barcode_band(image)):
                for decoder_name, runner in (
                    ("zxingcpp_bars", _decode_zxingcpp),
                    ("pyzbar_bars", _decode_pyzbar),
                    ("ean13_signal", _decode_signal),
                ):
                    attempt_id = f"{crop.crop_id}__v5_{decoder_name}__{variant_name}"
                    started = perf_counter()
                    try:
                        hits = runner(variant)
                    except Exception as exc:  # noqa: BLE001
                        attempts.append(
                            DecodeAttempt(
                                attempt_id=attempt_id,
                                crop_id=crop.crop_id,
                                decoder=f"v5_{decoder_name}",
                                variant=variant_name,
                                success=False,
                                error=str(exc),
                                duration_ms=_duration_ms(started),
                            )
                        )
                        warnings.append(
                            f"v5 barcode bars decoder {decoder_name} failed for "
                            f"{crop.crop_id}/{variant_name}: {exc}"
                        )
                        continue
                    attempts.append(
                        DecodeAttempt(
                            attempt_id=attempt_id,
                            crop_id=crop.crop_id,
                            decoder=f"v5_{decoder_name}",
                            variant=variant_name,
                            success=bool(hits),
                            duration_ms=_duration_ms(started),
                        )
                    )
                    for payload, confidence in hits:
                        digits = re.sub(r"\D+", "", payload)
                        if not _ean13_ok(digits):
                            continue
                        if float(confidence) > hit_confidence:
                            hit_value = digits
                            hit_decoder = f"v5_{decoder_name}"
                            hit_variant = variant_name
                            hit_confidence = float(confidence)
                    if hit_confidence >= config.min_confidence:
                        break
                if hit_confidence >= config.min_confidence:
                    break

            if not hit_value:
                continue
            key = (crop.detection_id, hit_value)
            if key in seen:
                continue
            seen.add(key)
            tracks_with_hit.add(track_id)
            symbols.append(
                DecodedSymbol(
                    symbol_id=f"{crop.crop_id}__v5_barcode_bars",
                    crop_id=crop.crop_id,
                    detection_id=crop.detection_id,
                    frame_index=crop.frame_index,
                    timestamp_ms=crop.timestamp_ms,
                    symbol_type="barcode",
                    decoder=hit_decoder,
                    variant=hit_variant,
                    payload=hit_value,
                    confidence=round(hit_confidence, 4),
                    attributes={
                        "track_id": track_id,
                        "source": "v5_crop_barcode_bars",
                        "crop_quality_score": crop.quality.score,
                    },
                )
            )

        context.decode_attempts.extend(attempts)
        context.decoded_symbols.extend(symbols)
        summary = {
            "enabled": True,
            "crops_processed": crops_processed,
            "crops_skipped_by_track_cap": skipped_by_track_cap,
            "crops_skipped_by_code_evidence": skipped_by_code_evidence,
            "tracks_skipped_by_code_evidence": len(coded_tracks),
            "decode_attempts": len(attempts),
            "hits": len(symbols),
            "tracks_with_hit": len(tracks_with_hit),
        }
        context.artifacts["v5_barcode_bars_decode"] = summary
        if not symbols:
            warnings.append("v5 crop barcode-bars produced no validated EAN-13 hits.")
        return StageOutcome(output_summary=summary, warnings=warnings)


def _track_id(crop: Any) -> str:
    value = crop.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return crop.detection_id


def _track_id_from_symbol(symbol: DecodedSymbol) -> str:
    value = symbol.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return symbol.detection_id


def _ean_coded_tracks(symbols: list[DecodedSymbol]) -> set[str]:
    return {
        _track_id_from_symbol(symbol)
        for symbol in symbols
        if _trusted_ean13_from_payload(symbol.payload)
    }


def _trusted_ean13_from_payload(payload: str) -> str:
    parsed = parse_qr_payload(str(payload or ""))
    values = [parsed.get("barcode", ""), parsed.get("qr_code_barcode", "")]
    values.append(str(payload or ""))
    for value in values:
        code = to_ean13(re.sub(r"\D+", "", value))
        if code and _v5_ean13_ok(code):
            return code
    return ""


def _has_code_evidence(symbols: list[DecodedSymbol]) -> bool:
    return any(_trusted_ean13_from_payload(symbol.payload) for symbol in symbols)


def _cap_crops_per_track(
    crops: list[Any],
    *,
    max_total: int,
    max_per_track: int,
    skip_track_ids: set[str] | None = None,
) -> list[Any]:
    skip_track_ids = skip_track_ids or set()
    selected: list[Any] = []
    by_track: dict[str, int] = {}
    for crop in crops:
        if len(selected) >= max_total:
            break
        track_id = _track_id(crop)
        if track_id in skip_track_ids:
            continue
        if max_per_track > 0:
            seen = by_track.get(track_id, 0)
            if seen >= max_per_track:
                continue
            by_track[track_id] = seen + 1
        selected.append(crop)
    return selected


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed
