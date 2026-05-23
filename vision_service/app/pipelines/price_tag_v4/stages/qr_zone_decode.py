from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import (
    BarcodeQrDecodeConfig,
    DecoderHit,
    OPTIONAL_PYZBAR_DECODER,
    OPTIONAL_ZXINGCPP_DECODER,
    OPTIONAL_ZXINGCPP_QR_ONLY_DECODER,
    _build_frame_lookup,
    _load_crop_image,
    _resolve_decoder_runners,
)
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.schemas.detections import DecodedSymbol
from app.utils.decoding import (
    DecodeVariantImage,
    build_decode_variants,
    normalize_decoded_payload,
)


@dataclass(frozen=True, slots=True)
class QrZoneDecodeConfig:
    enabled: bool
    max_crops: int
    min_crop_quality_score: float
    max_crops_per_track: int
    skip_if_code_evidence_present: bool
    stop_after_first_success_per_crop: bool
    enabled_zones: tuple[str, ...]
    enabled_variants: tuple[str, ...]


class QrZoneDecodeStage(BaseStage):
    name = "QrZoneDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops": config.max_crops,
            "min_crop_quality_score": config.min_crop_quality_score,
            "max_crops_per_track": config.max_crops_per_track,
            "skip_if_code_evidence_present": config.skip_if_code_evidence_present,
            "enabled_zones": config.enabled_zones,
            "enabled_variants": config.enabled_variants,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled:
            return StageOutcome(output_summary={"enabled": False, "decoded_symbols_count": 0})

        barcode_config = BarcodeQrDecodeConfig.from_context(context)
        warnings = list(barcode_config.warnings)
        decoder_runners = _qr_zone_decoder_runners(
            _resolve_decoder_runners(config=barcode_config, warnings=warnings)
        )
        if not decoder_runners or not context.crop_candidates:
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "crops_processed": 0,
                    "zone_attempts": 0,
                    "decoded_symbols_count": 0,
                },
                warnings=warnings,
            )

        existing_keys = {
            (symbol.detection_id, symbol.payload, symbol.symbol_type)
            for symbol in context.decoded_symbols
        }
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
            _coded_tracks(context.decoded_symbols)
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

        added: list[DecodedSymbol] = []
        crops_processed = 0
        zone_attempts = 0
        decode_attempts = 0

        for crop in eligible:
            image = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
            if image is None:
                continue
            crops_processed += 1
            symbol_rank = 0
            crop_had_hit = False
            for zone_name, zone in qr_zones(image, enabled_zones=config.enabled_zones):
                zone_attempts += 1
                variants = _filter_variants(
                    _qr_zone_variants(zone),
                    enabled_variants=config.enabled_variants,
                )
                for decoder_name, runner in decoder_runners:
                    for variant_name, variant in variants.items():
                        decode_attempts += 1
                        try:
                            hits = runner(variant)
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(
                                f"QR zone decoder {decoder_name} failed for {crop.crop_id} "
                                f"{zone_name}/{variant_name}: {exc}"
                            )
                            continue
                        for hit in hits:
                            payload = normalize_decoded_payload(hit.payload)
                            if not payload:
                                continue
                            symbol_type = hit.symbol_type or "qr"
                            key = (crop.detection_id, payload, symbol_type)
                            if key in existing_keys:
                                continue
                            existing_keys.add(key)
                            symbol_rank += 1
                            crop_had_hit = True
                            added.append(
                                DecodedSymbol(
                                    symbol_id=(
                                        f"{crop.crop_id}__qr_zone_{len(added) + 1:04d}_"
                                        f"{symbol_rank:02d}"
                                    ),
                                    crop_id=crop.crop_id,
                                    detection_id=crop.detection_id,
                                    frame_index=crop.frame_index,
                                    timestamp_ms=crop.timestamp_ms,
                                    symbol_type=symbol_type,
                                    decoder=f"qr_zone:{decoder_name}",
                                    variant=f"{zone_name}:{variant_name}",
                                    payload=payload,
                                    confidence=min(float(hit.confidence) + 0.03, 0.99),
                                    bbox=None,
                                    attributes={
                                        **(hit.attributes or {}),
                                        "track_id": crop.attributes.get("track_id", ""),
                                        "crop_quality_score": crop.quality.score,
                                        "zone": zone_name,
                                    },
                                )
                            )
                    if crop_had_hit and config.stop_after_first_success_per_crop:
                        break
                if crop_had_hit and config.stop_after_first_success_per_crop:
                    break

        context.decoded_symbols.extend(added)
        return StageOutcome(
            output_summary={
                "enabled": True,
                "crops_processed": crops_processed,
                "crops_skipped_by_track_cap": skipped_by_track_cap,
                "crops_skipped_by_code_evidence": skipped_by_code_evidence,
                "tracks_skipped_by_code_evidence": len(coded_tracks),
                "zone_attempts": zone_attempts,
                "decode_attempts": decode_attempts,
                "decoded_symbols_count": len(added),
                "total_decoded_symbols_count": len(context.decoded_symbols),
                "enabled_zones": config.enabled_zones,
                "enabled_variants": config.enabled_variants,
            },
            warnings=warnings,
        )


def config_from_context(context: PipelineContext) -> QrZoneDecodeConfig:
    raw = context.config.get("barcode_qr_zone_decode", {})
    if not isinstance(raw, dict):
        raw = {}
    return QrZoneDecodeConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        max_crops=positive_int(raw.get("max_crops"), default=240),
        min_crop_quality_score=float(raw.get("min_crop_quality_score", 0.05)),
        max_crops_per_track=non_negative_int(raw.get("max_crops_per_track"), default=0),
        skip_if_code_evidence_present=bool_value(
            raw.get("skip_if_code_evidence_present"),
            default=True,
        ),
        stop_after_first_success_per_crop=bool_value(
            raw.get("stop_after_first_success_per_crop"),
            default=True,
        ),
        enabled_zones=string_tuple(raw.get("enabled_zones")),
        enabled_variants=string_tuple(raw.get("enabled_variants")),
    )


def _coded_tracks(symbols: list[DecodedSymbol]) -> set[str]:
    return {
        _track_id_from_symbol(symbol)
        for symbol in symbols
        if _symbol_has_qr_or_trusted_ean(symbol)
    }


def _symbol_has_qr_or_trusted_ean(symbol: DecodedSymbol) -> bool:
    if _trusted_ean13_from_payload(symbol.payload):
        return True
    if symbol.symbol_type != "qr":
        return False
    parsed = parse_qr_payload(str(symbol.payload or ""))
    return any(field not in {"barcode", "qr_code_barcode"} for field in parsed)


def _trusted_ean13_from_payload(payload: str) -> str:
    parsed = parse_qr_payload(str(payload or ""))
    values = [parsed.get("barcode", ""), parsed.get("qr_code_barcode", "")]
    values.append(str(payload or ""))
    for value in values:
        code = _to_ean13(re.sub(r"\D+", "", value))
        if code and _ean13_ok(code):
            return code
    return ""


def _to_ean13(value: str) -> str:
    digits = re.sub(r"\D+", "", value or "")
    if len(digits) == 13:
        return digits
    if len(digits) == 12:
        return "0" + digits
    return ""


def _ean13_ok(value: str) -> bool:
    if not re.fullmatch(r"\d{13}", value or ""):
        return False
    checksum = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(value[:12]))
    return (10 - checksum % 10) % 10 == int(value[12])


def _track_id_from_symbol(symbol: DecodedSymbol) -> str:
    value = symbol.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return symbol.detection_id


def _track_id(crop: Any) -> str:
    value = crop.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return crop.detection_id


def _has_code_evidence(symbols: list[DecodedSymbol]) -> bool:
    return any(_symbol_has_qr_or_trusted_ean(symbol) for symbol in symbols)


def qr_zones(
    image: Any,
    *,
    enabled_zones: tuple[str, ...] = (),
) -> list[tuple[str, Any]]:
    h, w = image.shape[:2]
    boxes = {
        "qr_square": (0.62, 0.02, 0.98, 0.45),
        "upper_right": (0.48, 0.00, 1.00, 0.52),
        "right_mid": (0.52, 0.05, 1.00, 0.70),
        "qr_tight": (0.58, 0.00, 1.00, 0.50),
        "upper_left": (0.00, 0.00, 0.48, 0.52),
        "upper_band": (0.00, 0.00, 1.00, 0.48),
    }
    allowed_zones = set(enabled_zones)
    zones: list[tuple[str, Any]] = []
    for name, (x1f, y1f, x2f, y2f) in boxes.items():
        if allowed_zones and name not in allowed_zones:
            continue
        x1 = max(int(round(x1f * w)), 0)
        y1 = max(int(round(y1f * h)), 0)
        x2 = min(int(round(x2f * w)), w)
        y2 = min(int(round(y2f * h)), h)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        zone = image[y1:y2, x1:x2]
        zones.append(
            (
                name,
                cv2.copyMakeBorder(
                    zone,
                    20,
                    20,
                    20,
                    20,
                    cv2.BORDER_CONSTANT,
                    value=_white_border_value(zone),
                ),
            )
        )
    return zones


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


def _qr_zone_variants(zone: Any) -> dict[str, DecodeVariantImage]:
    variants = build_decode_variants(
        zone,
        include_original=True,
        include_grayscale=True,
        include_clahe=False,
        include_sharpened=False,
        include_resized_x2=True,
        include_resized_x3=False,
        include_resized_x4=False,
        include_adaptive_threshold=True,
        include_right_angle_rotations=False,
        include_small_angle_rotations=False,
    )
    variants.update(_targeted_qr_zone_variants(zone))
    return variants


def _filter_variants(
    variants: dict[str, DecodeVariantImage],
    *,
    enabled_variants: tuple[str, ...],
) -> dict[str, DecodeVariantImage]:
    allowed_variants = set(enabled_variants)
    if not allowed_variants:
        return variants
    return {
        name: variant
        for name, variant in variants.items()
        if name in allowed_variants
    }


def _targeted_qr_zone_variants(zone: Any) -> dict[str, DecodeVariantImage]:
    """High-res QR variants proven on tilted shelf crops."""
    if zone.size == 0:
        return {}
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if zone.ndim == 3 else zone
    gray_x4 = cv2.resize(
        gray,
        (max(gray.shape[1] * 4, 1), max(gray.shape[0] * 4, 1)),
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(gray_x4, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray_x4, 1.7, blurred, -0.7, 0)
    _, otsu = cv2.threshold(gray_x4, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return {
        "x4_gray": DecodeVariantImage(
            name="x4_gray",
            image=gray_x4,
            scale_x=4.0,
            scale_y=4.0,
        ),
        "x4_sharp": DecodeVariantImage(
            name="x4_sharp",
            image=sharp,
            scale_x=4.0,
            scale_y=4.0,
        ),
        "x4_otsu": DecodeVariantImage(
            name="x4_otsu",
            image=otsu,
            scale_x=4.0,
            scale_y=4.0,
        ),
    }


def _white_border_value(zone: Any) -> int | tuple[int, int, int]:
    if getattr(zone, "ndim", 0) == 2:
        return 255
    return (255, 255, 255)


def _qr_zone_decoder_runners(decoder_runners: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    has_qr_only_zxingcpp = any(
        name == OPTIONAL_ZXINGCPP_QR_ONLY_DECODER for name, _ in decoder_runners
    )
    runners: list[tuple[str, Any]] = []
    for name, runner in decoder_runners:
        if name == OPTIONAL_PYZBAR_DECODER:
            continue
        if has_qr_only_zxingcpp and name == OPTIONAL_ZXINGCPP_DECODER:
            continue
        runners.append(
            (
                name,
                _decode_qr_only_zxingcpp if name == OPTIONAL_ZXINGCPP_DECODER else runner,
            )
        )
    return runners


def _decode_qr_only_zxingcpp(variant: DecodeVariantImage) -> list[DecoderHit]:
    import zxingcpp  # type: ignore[import-not-found]

    hits: list[DecoderHit] = []
    results = zxingcpp.read_barcodes(
        variant.image,
        formats=zxingcpp.BarcodeFormat.QRCode,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
    )
    if not results and variant.name in {"x4_gray", "x4_sharp", "x4_otsu"}:
        results = zxingcpp.read_barcodes(
            variant.image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            is_pure=True,
        )
    for result in results:
        payload = normalize_decoded_payload(getattr(result, "text", ""))
        if not payload:
            continue
        hits.append(
            DecoderHit(
                payload=payload,
                symbol_type="qr",
                confidence=0.92,
                bbox=None,
                attributes={"format": "qr_code", "qr_zone_only": True},
            )
        )
    return hits


def bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def non_negative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list | tuple | set):
        raw_items = list(value)
    else:
        return ()
    return tuple(
        item.strip()
        for item in (str(raw_item) for raw_item in raw_items)
        if item.strip()
    )
