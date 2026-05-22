from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import (
    BarcodeQrDecodeConfig,
    DecoderHit,
    OPTIONAL_PYZBAR_DECODER,
    OPTIONAL_ZXINGCPP_DECODER,
    _build_frame_lookup,
    _load_crop_image,
    _resolve_decoder_runners,
)
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
    stop_after_first_success_per_crop: bool


class QrZoneDecodeStage(BaseStage):
    name = "QrZoneDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops": config.max_crops,
            "min_crop_quality_score": config.min_crop_quality_score,
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
        eligible = [
            crop
            for crop in sorted(
                context.crop_candidates,
                key=lambda item: item.quality.score,
                reverse=True,
            )
            if crop.quality.score >= config.min_crop_quality_score
        ][: config.max_crops]

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
            for zone_name, zone in qr_zones(image):
                zone_attempts += 1
                variants = _qr_zone_variants(zone)
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
                "zone_attempts": zone_attempts,
                "decode_attempts": decode_attempts,
                "decoded_symbols_count": len(added),
                "total_decoded_symbols_count": len(context.decoded_symbols),
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
        stop_after_first_success_per_crop=bool_value(
            raw.get("stop_after_first_success_per_crop"),
            default=True,
        ),
    )


def qr_zones(image: Any) -> list[tuple[str, Any]]:
    h, w = image.shape[:2]
    boxes = {
        "qr_square": (0.62, 0.02, 0.98, 0.45),
        "upper_right": (0.48, 0.00, 1.00, 0.52),
        "right_mid": (0.52, 0.05, 1.00, 0.70),
        "qr_tight": (0.58, 0.00, 1.00, 0.50),
        "upper_left": (0.00, 0.00, 0.48, 0.52),
        "upper_band": (0.00, 0.00, 1.00, 0.48),
    }
    zones: list[tuple[str, Any]] = []
    for name, (x1f, y1f, x2f, y2f) in boxes.items():
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
    return [
        (name, _decode_qr_only_zxingcpp if name == OPTIONAL_ZXINGCPP_DECODER else runner)
        for name, runner in decoder_runners
        if name != OPTIONAL_PYZBAR_DECODER
    ]


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
