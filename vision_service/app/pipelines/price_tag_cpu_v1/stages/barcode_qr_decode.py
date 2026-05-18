from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, SampledFrameMetadata, StageOutcome
from app.schemas.detections import BoundingBox, CropCandidate, DecodeAttempt, DecodedSymbol
from app.utils.decoding import DecodeVariantImage, build_decode_variants, normalize_decoded_payload
from app.utils.image_processing import clip_bbox_to_frame

OPENCV_QR_DECODER = "opencv_qr_detector"
OPTIONAL_ZXINGCPP_DECODER = "zxingcpp"
OPTIONAL_PYZBAR_DECODER = "pyzbar"
OPTIONAL_ARUCO_DECODER = "aruco_qr"


@dataclass(slots=True)
class BarcodeQrDecodeConfig:
    enabled: bool
    max_crops: int
    min_crop_quality_score: float
    opencv_qr_detector_enabled: bool
    optional_zxingcpp_enabled: bool
    optional_pyzbar_enabled: bool
    optional_aruco_enabled: bool
    variants: dict[str, bool]
    stop_after_first_success_per_crop: bool
    max_payload_preview_length: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "BarcodeQrDecodeConfig":
        raw_config = context.config.get("barcode_qr_decode", {})
        warnings: list[str] = []

        if not isinstance(raw_config, dict):
            warnings.append(
                "Barcode/QR decode config must be a mapping. Falling back to defaults."
            )
            raw_config = {}

        raw_decoders = raw_config.get("decoders", {})
        if not isinstance(raw_decoders, dict):
            warnings.append("barcode_qr_decode.decoders must be a mapping.")
            raw_decoders = {}

        raw_variants = raw_config.get("variants", {})
        if not isinstance(raw_variants, dict):
            warnings.append("barcode_qr_decode.variants must be a mapping.")
            raw_variants = {}

        return cls(
            enabled=_coerce_bool(
                raw_config.get("enabled"),
                default=True,
                field_name="enabled",
                warnings=warnings,
            ),
            max_crops=_coerce_positive_int(
                raw_config.get("max_crops"),
                default=100,
                field_name="max_crops",
                warnings=warnings,
            ),
            min_crop_quality_score=_coerce_ratio_inclusive_zero(
                raw_config.get("min_crop_quality_score"),
                default=0.15,
                field_name="min_crop_quality_score",
                warnings=warnings,
            ),
            opencv_qr_detector_enabled=_decode_enabled_flag(
                raw_decoders.get(OPENCV_QR_DECODER),
                default=True,
                field_name=OPENCV_QR_DECODER,
                warnings=warnings,
            ),
            optional_zxingcpp_enabled=_decode_enabled_flag(
                raw_decoders.get("optional_zxingcpp"),
                default=False,
                field_name="optional_zxingcpp",
                warnings=warnings,
            ),
            optional_pyzbar_enabled=_decode_enabled_flag(
                raw_decoders.get("optional_pyzbar"),
                default=False,
                field_name="optional_pyzbar",
                warnings=warnings,
            ),
            optional_aruco_enabled=_decode_enabled_flag(
                raw_decoders.get("optional_aruco"),
                default=False,
                field_name="optional_aruco",
                warnings=warnings,
            ),
            variants={
                "original": _coerce_bool(
                    raw_variants.get("original"),
                    default=True,
                    field_name="variants.original",
                    warnings=warnings,
                ),
                "grayscale": _coerce_bool(
                    raw_variants.get("grayscale"),
                    default=True,
                    field_name="variants.grayscale",
                    warnings=warnings,
                ),
                "clahe": _coerce_bool(
                    raw_variants.get("clahe"),
                    default=True,
                    field_name="variants.clahe",
                    warnings=warnings,
                ),
                "sharpened": _coerce_bool(
                    raw_variants.get("sharpened"),
                    default=True,
                    field_name="variants.sharpened",
                    warnings=warnings,
                ),
                "resized_x2": _coerce_bool(
                    raw_variants.get("resized_x2"),
                    default=True,
                    field_name="variants.resized_x2",
                    warnings=warnings,
                ),
                "resized_x3": _coerce_bool(
                    raw_variants.get("resized_x3"),
                    default=False,
                    field_name="variants.resized_x3",
                    warnings=warnings,
                ),
                "resized_x4": _coerce_bool(
                    raw_variants.get("resized_x4"),
                    default=False,
                    field_name="variants.resized_x4",
                    warnings=warnings,
                ),
                "adaptive_threshold": _coerce_bool(
                    raw_variants.get("adaptive_threshold"),
                    default=True,
                    field_name="variants.adaptive_threshold",
                    warnings=warnings,
                ),
                "right_angle_rotations": _coerce_bool(
                    raw_variants.get("right_angle_rotations"),
                    default=False,
                    field_name="variants.right_angle_rotations",
                    warnings=warnings,
                ),
                "small_angle_rotations": _coerce_bool(
                    raw_variants.get("small_angle_rotations"),
                    default=False,
                    field_name="variants.small_angle_rotations",
                    warnings=warnings,
                ),
            },
            stop_after_first_success_per_crop=_coerce_bool(
                raw_config.get("stop_after_first_success_per_crop"),
                default=False,
                field_name="stop_after_first_success_per_crop",
                warnings=warnings,
            ),
            max_payload_preview_length=_coerce_positive_int(
                raw_config.get("max_payload_preview_length"),
                default=300,
                field_name="max_payload_preview_length",
                warnings=warnings,
            ),
            warnings=warnings,
        )

    def enabled_decoders(self) -> list[str]:
        decoders: list[str] = []
        if self.opencv_qr_detector_enabled:
            decoders.append(OPENCV_QR_DECODER)
        if self.optional_zxingcpp_enabled:
            decoders.append(OPTIONAL_ZXINGCPP_DECODER)
        if self.optional_pyzbar_enabled:
            decoders.append(OPTIONAL_PYZBAR_DECODER)
        if self.optional_aruco_enabled:
            decoders.append(OPTIONAL_ARUCO_DECODER)
        return decoders


@dataclass(slots=True)
class DecoderHit:
    payload: str
    symbol_type: str
    confidence: float
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] | None = None


class BarcodeQrDecodeStage(BaseStage):
    name = "BarcodeQrDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = BarcodeQrDecodeConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops": config.max_crops,
            "min_crop_quality_score": config.min_crop_quality_score,
            "enabled_decoders": config.enabled_decoders(),
            "enabled_variants": [
                name for name, is_enabled in config.variants.items() if is_enabled
            ],
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = BarcodeQrDecodeConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            context.decode_attempts = []
            context.decoded_symbols = []
            return StageOutcome(
                output_summary={
                    "enabled": False,
                    "decode_attempts_count": 0,
                    "decoded_symbols_count": 0,
                    "decoded_symbols_by_type": context.decoded_symbols_by_type(),
                },
                warnings=warnings,
            )

        if not context.crop_candidates:
            context.decode_attempts = []
            context.decoded_symbols = []
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "crops_processed": 0,
                    "decode_attempts_count": 0,
                    "decoded_symbols_count": 0,
                    "decoded_symbols_by_type": context.decoded_symbols_by_type(),
                },
                warnings=warnings,
            )

        decoder_runners = _resolve_decoder_runners(config=config, warnings=warnings)
        if not decoder_runners:
            context.decode_attempts = []
            context.decoded_symbols = []
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "crops_processed": 0,
                    "decode_attempts_count": 0,
                    "decoded_symbols_count": 0,
                    "decoded_symbols_by_type": context.decoded_symbols_by_type(),
                },
                warnings=warnings,
            )

        frame_lookup = _build_frame_lookup(context.sampled_frames)
        attempts: list[DecodeAttempt] = []
        decoded_symbols: list[DecodedSymbol] = []

        eligible_crops = [
            crop
            for crop in sorted(
                context.crop_candidates,
                key=lambda item: item.quality.score,
                reverse=True,
            )
            if crop.quality.score >= config.min_crop_quality_score
        ][: config.max_crops]

        crops_processed = 0
        skipped_low_quality = max(len(context.crop_candidates) - len(eligible_crops), 0)

        for crop in eligible_crops:
            crop_image = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
            if crop_image is None:
                warnings.append(
                    f"Crop {crop.crop_id} could not be reconstructed from sampled frames."
                )
                continue

            variants = build_decode_variants(
                crop_image,
                include_original=config.variants["original"],
                include_grayscale=config.variants["grayscale"],
                include_clahe=config.variants["clahe"],
                include_sharpened=config.variants["sharpened"],
                include_resized_x2=config.variants["resized_x2"],
                include_resized_x3=config.variants["resized_x3"],
                include_resized_x4=config.variants["resized_x4"],
                include_adaptive_threshold=config.variants["adaptive_threshold"],
                include_right_angle_rotations=config.variants["right_angle_rotations"],
                include_small_angle_rotations=config.variants["small_angle_rotations"],
            )
            seen_payloads: set[tuple[str, str]] = set()
            crops_processed += 1
            stop_crop_processing = False
            symbol_rank = 0

            for decoder_name, runner in decoder_runners:
                for variant_name, variant in variants.items():
                    attempt_id = f"{crop.crop_id}__{decoder_name}__{variant_name}"
                    started = perf_counter()

                    try:
                        hits = runner(variant)
                    except Exception as exc:
                        attempts.append(
                            DecodeAttempt(
                                attempt_id=attempt_id,
                                crop_id=crop.crop_id,
                                decoder=decoder_name,
                                variant=variant_name,
                                success=False,
                                error=str(exc),
                                duration_ms=_duration_ms(started),
                            )
                        )
                        warnings.append(
                            f"Decoder {decoder_name} failed for crop {crop.crop_id} "
                            f"variant {variant_name}: {exc}"
                        )
                        continue

                    valid_hits: list[DecoderHit] = []
                    for hit in hits:
                        payload = normalize_decoded_payload(hit.payload)
                        if not payload:
                            continue
                        valid_hits.append(
                            DecoderHit(
                                payload=payload,
                                symbol_type=hit.symbol_type,
                                confidence=hit.confidence,
                                bbox=hit.bbox,
                                attributes=hit.attributes or {},
                            )
                        )

                    attempts.append(
                        DecodeAttempt(
                            attempt_id=attempt_id,
                            crop_id=crop.crop_id,
                            decoder=decoder_name,
                            variant=variant_name,
                            success=bool(valid_hits),
                            duration_ms=_duration_ms(started),
                        )
                    )

                    new_symbol_added = False
                    for hit in valid_hits:
                        symbol_key = (hit.payload, hit.symbol_type)
                        if symbol_key in seen_payloads:
                            continue
                        seen_payloads.add(symbol_key)
                        symbol_rank += 1
                        decoded_symbols.append(
                            DecodedSymbol(
                                symbol_id=f"{crop.crop_id}__symbol_{symbol_rank:02d}",
                                crop_id=crop.crop_id,
                                detection_id=crop.detection_id,
                                frame_index=crop.frame_index,
                                timestamp_ms=crop.timestamp_ms,
                                symbol_type=hit.symbol_type,
                                decoder=decoder_name,
                                variant=variant_name,
                                payload=hit.payload,
                                confidence=round(hit.confidence, 4),
                                bbox=hit.bbox,
                                attributes={
                                    **hit.attributes,
                                    "crop_quality_score": crop.quality.score,
                                },
                            )
                        )
                        new_symbol_added = True

                    if config.stop_after_first_success_per_crop and new_symbol_added:
                        stop_crop_processing = True
                        break

                if stop_crop_processing:
                    break

        context.decode_attempts = attempts
        context.decoded_symbols = decoded_symbols

        return StageOutcome(
            output_summary={
                "enabled": True,
                "crops_processed": crops_processed,
                "crops_skipped_low_quality": skipped_low_quality,
                "decode_attempts_count": len(attempts),
                "decoded_symbols_count": len(decoded_symbols),
                "decoded_symbols_by_type": context.decoded_symbols_by_type(),
            },
            warnings=warnings,
        )


def _resolve_decoder_runners(
    *,
    config: BarcodeQrDecodeConfig,
    warnings: list[str],
) -> list[tuple[str, Callable[[DecodeVariantImage], list[DecoderHit]]]]:
    runners: list[tuple[str, Callable[[DecodeVariantImage], list[DecoderHit]]]] = []

    if config.opencv_qr_detector_enabled:
        qr_detector = cv2.QRCodeDetector()
        runners.append(
            (
                OPENCV_QR_DECODER,
                lambda variant, detector=qr_detector: _decode_with_opencv_qr_detector(
                    detector=detector,
                    variant=variant,
                ),
            )
        )

    if config.optional_zxingcpp_enabled:
        if find_spec("zxingcpp") is None:
            warnings.append(
                "optional_zxingcpp decoder was enabled in config but zxingcpp is not installed."
            )
        else:
            runners.append(
                (
                    OPTIONAL_ZXINGCPP_DECODER,
                    _decode_with_optional_zxingcpp,
                )
            )

    if config.optional_pyzbar_enabled:
        if find_spec("pyzbar") is None:
            warnings.append(
                "optional_pyzbar decoder was enabled in config but pyzbar is not installed."
            )
        else:
            try:
                from pyzbar import pyzbar as pyzbar_module  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    "optional_pyzbar decoder was enabled but could not be initialized: "
                    f"{exc}"
                )
            else:
                runners.append(
                    (
                        OPTIONAL_PYZBAR_DECODER,
                        lambda variant, module=pyzbar_module: _decode_with_optional_pyzbar(
                            variant=variant,
                            pyzbar_module=module,
                        ),
                    )
                )

    if config.optional_aruco_enabled:
        aruco_detector = _build_aruco_detector()
        if aruco_detector is None:
            warnings.append(
                "optional_aruco decoder was enabled but cv2.QRCodeDetectorAruco "
                "is unavailable in this OpenCV build."
            )
        else:
            runners.append(
                (
                    OPTIONAL_ARUCO_DECODER,
                    lambda variant, detector=aruco_detector: _decode_with_aruco_qr_detector(
                        detector=detector,
                        variant=variant,
                    ),
                )
            )

    return runners


def _build_aruco_detector() -> Any | None:
    factory = getattr(cv2, "QRCodeDetectorAruco", None)
    if factory is None:
        return None
    try:
        return factory()
    except Exception:  # noqa: BLE001
        return None


def _decode_with_aruco_qr_detector(
    *,
    detector: Any,
    variant: DecodeVariantImage,
) -> list[DecoderHit]:
    """ArUco-based QR finder; better on tilted / small 4K QR codes."""
    hits: list[DecoderHit] = []
    if hasattr(detector, "detectAndDecodeMulti"):
        multi_payloads, multi_points = _detect_and_decode_multi(detector, variant.image)
        for payload, points in zip(multi_payloads, multi_points):
            hits.append(
                DecoderHit(
                    payload=payload,
                    symbol_type="qr",
                    confidence=_decoder_confidence(
                        decoder=OPTIONAL_ARUCO_DECODER, variant=variant.name
                    ),
                    bbox=_bbox_from_points(points=points, variant=variant),
                    attributes={"multi": True},
                )
            )
    if hits:
        return hits

    payload, points = _detect_and_decode_single(detector, variant.image)
    if not payload:
        return []
    return [
        DecoderHit(
            payload=payload,
            symbol_type="qr",
            confidence=_decoder_confidence(
                decoder=OPTIONAL_ARUCO_DECODER, variant=variant.name
            ),
            bbox=_bbox_from_points(points=points, variant=variant),
            attributes={"multi": False},
        )
    ]


def _build_frame_lookup(
    sampled_frames: list[SampledFrameMetadata],
) -> dict[int, SampledFrameMetadata]:
    return {frame.frame_index: frame for frame in sampled_frames}


def _load_crop_image(
    *,
    crop: CropCandidate,
    frame_lookup: dict[int, SampledFrameMetadata],
) -> np.ndarray | None:
    frame_meta = frame_lookup.get(crop.frame_index)
    if frame_meta is None or frame_meta.local_frame_path is None:
        return None

    frame_path = Path(frame_meta.local_frame_path)
    if not frame_path.exists():
        return None

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None

    padded_bbox = clip_bbox_to_frame(
        crop.padded_bbox,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )
    if padded_bbox is None:
        return None

    crop_image = frame[
        padded_bbox.y_min : padded_bbox.y_max,
        padded_bbox.x_min : padded_bbox.x_max,
    ]
    if crop_image.size == 0:
        return None
    return crop_image


def _decode_with_opencv_qr_detector(
    *,
    detector: cv2.QRCodeDetector,
    variant: DecodeVariantImage,
) -> list[DecoderHit]:
    hits: list[DecoderHit] = []

    if hasattr(detector, "detectAndDecodeMulti"):
        multi_payloads, multi_points = _detect_and_decode_multi(detector, variant.image)
        for payload, points in zip(multi_payloads, multi_points):
            hits.append(
                DecoderHit(
                    payload=payload,
                    symbol_type="qr",
                    confidence=_decoder_confidence(decoder=OPENCV_QR_DECODER, variant=variant.name),
                    bbox=_bbox_from_points(points=points, variant=variant),
                    attributes={"multi": True},
                )
            )

    if hits:
        return hits

    payload, points = _detect_and_decode_single(detector, variant.image)
    if not payload:
        return []

    return [
        DecoderHit(
            payload=payload,
            symbol_type="qr",
            confidence=_decoder_confidence(decoder=OPENCV_QR_DECODER, variant=variant.name),
            bbox=_bbox_from_points(points=points, variant=variant),
            attributes={"multi": False},
        )
    ]


def _detect_and_decode_single(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
) -> tuple[str, np.ndarray | None]:
    decoded_payload, points, _ = detector.detectAndDecode(image)
    return normalize_decoded_payload(decoded_payload), points


def _detect_and_decode_multi(
    detector: cv2.QRCodeDetector,
    image: np.ndarray,
) -> tuple[list[str], list[np.ndarray]]:
    decoded_payloads: list[str] = []
    decoded_points: list[np.ndarray] = []

    multi_result = detector.detectAndDecodeMulti(image)
    if not isinstance(multi_result, tuple) or len(multi_result) < 3:
        return decoded_payloads, decoded_points

    was_detected = bool(multi_result[0])
    payloads = multi_result[1] or []
    points = multi_result[2]
    if not was_detected or points is None:
        return decoded_payloads, decoded_points

    for index, payload in enumerate(payloads):
        normalized_payload = normalize_decoded_payload(payload)
        if not normalized_payload:
            continue
        decoded_payloads.append(normalized_payload)
        decoded_points.append(points[index])

    return decoded_payloads, decoded_points


def _decode_with_optional_zxingcpp(variant: DecodeVariantImage) -> list[DecoderHit]:
    import zxingcpp  # type: ignore[import-not-found]

    hits: list[DecoderHit] = []
    for result in zxingcpp.read_barcodes(variant.image):
        payload = normalize_decoded_payload(getattr(result, "text", ""))
        if not payload:
            continue

        format_name = str(getattr(result, "format", "unknown")).lower()
        symbol_type = "unknown"
        if "qr" in format_name:
            symbol_type = "qr"
        elif format_name != "unknown":
            symbol_type = "barcode"

        hits.append(
            DecoderHit(
                payload=payload,
                symbol_type=symbol_type,
                confidence=_decoder_confidence(decoder=OPTIONAL_ZXINGCPP_DECODER, variant=variant.name),
                bbox=None,
                attributes={"format": format_name},
            )
        )

    return hits


def _decode_with_optional_pyzbar(
    *,
    variant: DecodeVariantImage,
    pyzbar_module: Any,
) -> list[DecoderHit]:
    hits: list[DecoderHit] = []
    for result in pyzbar_module.decode(variant.image):
        payload = normalize_decoded_payload(getattr(result, "data", b""))
        if not payload:
            continue

        format_name = str(getattr(result, "type", "unknown")).lower()
        symbol_type = "unknown"
        if "qr" in format_name:
            symbol_type = "qr"
        elif format_name != "unknown":
            symbol_type = "barcode"

        hits.append(
            DecoderHit(
                payload=payload,
                symbol_type=symbol_type,
                confidence=_decoder_confidence(
                    decoder=OPTIONAL_PYZBAR_DECODER,
                    variant=variant.name,
                ),
                bbox=None,
                attributes={"format": format_name},
            )
        )

    return hits


def _bbox_from_points(
    *,
    points: np.ndarray | None,
    variant: DecodeVariantImage,
) -> BoundingBox | None:
    if points is None:
        return None
    if abs(variant.rotation_degrees) > 1e-6:
        return None

    normalized_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if normalized_points.size == 0:
        return None

    scaled_x = normalized_points[:, 0] / max(variant.scale_x, 1e-6)
    scaled_y = normalized_points[:, 1] / max(variant.scale_y, 1e-6)

    x_min = int(np.floor(np.min(scaled_x)))
    y_min = int(np.floor(np.min(scaled_y)))
    x_max = int(np.ceil(np.max(scaled_x)))
    y_max = int(np.ceil(np.max(scaled_y)))

    try:
        return BoundingBox(
            x_min=max(x_min, 0),
            y_min=max(y_min, 0),
            x_max=max(x_max, max(x_min, 0) + 1),
            y_max=max(y_max, max(y_min, 0) + 1),
        )
    except ValueError:
        return None


def _decoder_confidence(*, decoder: str, variant: str) -> float:
    base_confidence = 0.68 if decoder == OPENCV_QR_DECODER else 0.82
    if decoder == OPTIONAL_PYZBAR_DECODER:
        base_confidence = 0.78
    variant_bonus = {
        "original": 0.00,
        "grayscale": 0.02,
        "clahe": 0.05,
        "sharpened": 0.04,
        "resized_x2": 0.08,
        "resized_x3": 0.09,
        "resized_x4": 0.10,
        "adaptive_threshold": 0.03,
    }
    if variant.startswith("rotated_"):
        return float(min(base_confidence + 0.08, 0.95))
    if variant.startswith("tilt_"):
        return float(min(base_confidence + 0.04, 0.95))
    return float(min(base_confidence + variant_bonus.get(variant, 0.0), 0.95))


def _duration_ms(started: float) -> int:
    return max(int(round((perf_counter() - started) * 1000)), 0)


def _decode_enabled_flag(
    value: Any,
    *,
    default: bool,
    field_name: str,
    warnings: list[str],
) -> bool:
    if isinstance(value, dict):
        return _coerce_bool(
            value.get("enabled"),
            default=default,
            field_name=f"{field_name}.enabled",
            warnings=warnings,
        )

    if value is None:
        return default

    return _coerce_bool(
        value,
        default=default,
        field_name=field_name,
        warnings=warnings,
    )


def _coerce_bool(
    value: Any,
    *,
    default: bool,
    field_name: str,
    warnings: list[str],
) -> bool:
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
    warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
    return default


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    field_name: str,
    warnings: list[str],
) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0:
        warnings.append(f"{field_name} must be > 0. Falling back to {default}.")
        return default
    return coerced


def _coerce_ratio_inclusive_zero(
    value: Any,
    *,
    default: float,
    field_name: str,
    warnings: list[str],
) -> float:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced < 0 or coerced > 1:
        warnings.append(
            f"{field_name} must be within [0, 1]. Falling back to {default}."
        )
        return default
    return coerced
