from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.zonal_ocr import _TAG_ZONES, _zone_box
from app.schemas.detections import DecodedSymbol, DecodeAttempt
from app.utils.decoding import normalize_decoded_payload


@dataclass(slots=True)
class _BarcodeBarsConfig:
    enabled: bool
    max_tracks: int
    min_confidence: float


class BarcodeBarsDecodeStage(BaseStage):
    name = "BarcodeBarsDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        fused = context.artifacts.get("v3_fused_tracks", [])
        return {
            "enabled": config.enabled,
            "fused_tracks": len(fused) if isinstance(fused, list) else 0,
            "max_tracks": config.max_tracks,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        fused_tracks = context.artifacts.get("v3_fused_tracks", [])
        if not config.enabled or not isinstance(fused_tracks, list) or not fused_tracks:
            return StageOutcome(
                output_summary={"enabled": config.enabled, "tracks_processed": 0, "hits": 0}
            )

        attempts: list[DecodeAttempt] = []
        symbols: list[DecodedSymbol] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = {
            (symbol.detection_id, re.sub(r"\D+", "", symbol.payload))
            for symbol in context.decoded_symbols
        }
        tracks_processed = 0

        for fused in fused_tracks[: config.max_tracks]:
            image = _read_image(fused)
            if image is None:
                continue
            track_id = str(fused.get("track_id", ""))
            crop_id = str(fused.get("anchor_crop_id", ""))
            detection_id = str(fused.get("anchor_detection_id", ""))
            if not crop_id or not detection_id:
                continue
            tracks_processed += 1
            barcode_band = _barcode_band(image)
            variants = _barcode_variants(barcode_band)
            hit_value = ""
            hit_decoder = ""
            hit_variant = ""
            hit_confidence = 0.0

            for variant_name, variant in variants:
                for decoder_name, runner in (
                    ("zxingcpp_bars", _decode_zxingcpp),
                    ("pyzbar_bars", _decode_pyzbar),
                    ("ean13_signal", _decode_signal),
                ):
                    attempt_id = f"{crop_id}__{decoder_name}__{variant_name}"
                    started = perf_counter()
                    try:
                        hits = runner(variant)
                    except Exception as exc:  # noqa: BLE001
                        attempts.append(
                            DecodeAttempt(
                                attempt_id=attempt_id,
                                crop_id=crop_id,
                                decoder=decoder_name,
                                variant=variant_name,
                                success=False,
                                error=str(exc),
                                duration_ms=_duration_ms(started),
                            )
                        )
                        continue
                    attempts.append(
                        DecodeAttempt(
                            attempt_id=attempt_id,
                            crop_id=crop_id,
                            decoder=decoder_name,
                            variant=variant_name,
                            success=bool(hits),
                            duration_ms=_duration_ms(started),
                        )
                    )
                    for payload, confidence in hits:
                        digits = re.sub(r"\D+", "", payload)
                        if not _ean13_ok(digits):
                            continue
                        if confidence > hit_confidence:
                            hit_value = digits
                            hit_decoder = decoder_name
                            hit_variant = variant_name
                            hit_confidence = confidence
                    if hit_confidence >= config.min_confidence:
                        break
                if hit_confidence >= config.min_confidence:
                    break

            if not hit_value:
                continue
            key = (detection_id, hit_value)
            if key in seen:
                continue
            seen.add(key)
            symbols.append(
                DecodedSymbol(
                    symbol_id=f"{crop_id}__v3_barcode_bars",
                    crop_id=crop_id,
                    detection_id=detection_id,
                    frame_index=int(fused.get("anchor_frame_index", 0) or 0),
                    timestamp_ms=fused.get("anchor_timestamp_ms"),
                    symbol_type="barcode",
                    decoder=hit_decoder,
                    variant=hit_variant,
                    payload=hit_value,
                    confidence=round(float(hit_confidence), 4),
                    attributes={
                        "track_id": track_id,
                        "source": "fused_barcode_zone",
                        "debug_key": fused.get("debug_key", ""),
                    },
                )
            )

        context.decode_attempts.extend(attempts)
        context.decoded_symbols.extend(symbols)
        context.artifacts["barcode_bars_decode"] = {
            "enabled": True,
            "tracks_processed": tracks_processed,
            "decode_attempts": len(attempts),
            "hits": len(symbols),
        }
        if not symbols:
            warnings.append("Barcode-from-bars produced no validated EAN-13 hits.")
        return StageOutcome(
            output_summary=context.artifacts["barcode_bars_decode"],
            warnings=warnings,
        )


def _read_image(fused: dict[str, Any]) -> np.ndarray | None:
    for key in ("deblurred_local_path", "local_path"):
        path = Path(str(fused.get(key, "")))
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            return image
    return None


def _barcode_band(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _zone_box(_TAG_ZONES["barcode"], width, height)
    band = image[y1:y2, x1:x2]
    if band.size == 0:
        return image
    return band


def _barcode_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_x = max(2, int(round(1100 / max(gray.shape[1], 1))))
    scale_y = max(2, int(round(260 / max(gray.shape[0], 1))))
    resized = cv2.resize(
        gray,
        (gray.shape[1] * scale_x, gray.shape[0] * scale_y),
        interpolation=cv2.INTER_CUBIC,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    otsu = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return [
        ("gray_x", resized),
        ("clahe_x", clahe),
        ("otsu_x", otsu),
        ("adaptive_x", adaptive),
    ]


def _decode_zxingcpp(image: np.ndarray) -> list[tuple[str, float]]:
    try:
        import zxingcpp  # type: ignore[import-not-found]
    except Exception:
        return []
    hits: list[tuple[str, float]] = []
    for result in zxingcpp.read_barcodes(image):
        payload = normalize_decoded_payload(getattr(result, "text", ""))
        digits = re.sub(r"\D+", "", payload)
        if digits:
            hits.append((digits, 0.90))
    return hits


def _decode_pyzbar(image: np.ndarray) -> list[tuple[str, float]]:
    try:
        from pyzbar import pyzbar as pyzbar_module  # type: ignore[import-not-found]
    except Exception:
        return []
    hits: list[tuple[str, float]] = []
    for result in pyzbar_module.decode(image):
        payload = normalize_decoded_payload(getattr(result, "data", b""))
        digits = re.sub(r"\D+", "", payload)
        if digits:
            hits.append((digits, 0.84))
    return hits


_L_PATTERNS = {
    "0001101": "0",
    "0011001": "1",
    "0010011": "2",
    "0111101": "3",
    "0100011": "4",
    "0110001": "5",
    "0101111": "6",
    "0111011": "7",
    "0110111": "8",
    "0001011": "9",
}
_G_PATTERNS = {
    "0100111": "0",
    "0110011": "1",
    "0011011": "2",
    "0100001": "3",
    "0011101": "4",
    "0111001": "5",
    "0000101": "6",
    "0010001": "7",
    "0001001": "8",
    "0010111": "9",
}
_R_PATTERNS = {
    "1110010": "0",
    "1100110": "1",
    "1101100": "2",
    "1000010": "3",
    "1011100": "4",
    "1001110": "5",
    "1010000": "6",
    "1000100": "7",
    "1001000": "8",
    "1110100": "9",
}
_PARITY_TO_FIRST = {
    "LLLLLL": "0",
    "LLGLGG": "1",
    "LLGGLG": "2",
    "LLGGGL": "3",
    "LGLLGG": "4",
    "LGGLLG": "5",
    "LGGGLL": "6",
    "LGLGLG": "7",
    "LGLGGL": "8",
    "LGGLGL": "9",
}


def _decode_signal(image: np.ndarray) -> list[tuple[str, float]]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 4 or gray.shape[1] < 95:
        return []
    signal = 1.0 - (gray.astype(np.float32).mean(axis=0) / 255.0)
    signal = np.convolve(signal, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    candidates: list[tuple[str, float]] = []
    for threshold in (0.35, 0.42, 0.50, 0.58):
        binary = signal > threshold
        black = np.flatnonzero(binary)
        if len(black) < 30:
            continue
        left = int(black[0])
        right = int(black[-1])
        span = right - left + 1
        if span < 95:
            continue
        module_width = span / 95.0
        for offset in (-1.2, -0.6, 0.0, 0.6, 1.2):
            for scale in (0.96, 0.99, 1.0, 1.01, 1.04):
                start = left + (offset * module_width)
                bits = _sample_modules(binary, start=start, module_width=module_width * scale)
                decoded, error = _decode_ean13_bits(bits)
                if decoded and _ean13_ok(decoded):
                    confidence = max(0.72, 0.88 - (error * 0.025))
                    candidates.append((decoded, confidence))
    if not candidates:
        return []
    best = max(candidates, key=lambda item: item[1])
    return [best]


def _sample_modules(binary: np.ndarray, *, start: float, module_width: float) -> str:
    bits = []
    for module_index in range(95):
        center = int(round(start + (module_index + 0.5) * module_width))
        lo = max(int(round(center - module_width * 0.35)), 0)
        hi = min(int(round(center + module_width * 0.35)) + 1, len(binary))
        if hi <= lo:
            bits.append("0")
        else:
            bits.append("1" if float(binary[lo:hi].mean()) >= 0.5 else "0")
    return "".join(bits)


def _decode_ean13_bits(bits: str) -> tuple[str, int]:
    if len(bits) != 95:
        return "", 999
    error = _hamming(bits[0:3], "101") + _hamming(bits[45:50], "01010") + _hamming(bits[92:95], "101")
    digits = []
    parity = []
    for index in range(6):
        chunk = bits[3 + index * 7 : 10 + index * 7]
        l_digit, l_error = _nearest_pattern(chunk, _L_PATTERNS)
        g_digit, g_error = _nearest_pattern(chunk, _G_PATTERNS)
        if l_error <= g_error:
            digits.append(l_digit)
            parity.append("L")
            error += l_error
        else:
            digits.append(g_digit)
            parity.append("G")
            error += g_error
    first = _PARITY_TO_FIRST.get("".join(parity), "")
    if not first:
        return "", error + 20
    right_digits = []
    for index in range(6):
        chunk = bits[50 + index * 7 : 57 + index * 7]
        digit, digit_error = _nearest_pattern(chunk, _R_PATTERNS)
        right_digits.append(digit)
        error += digit_error
    if error > 8:
        return "", error
    return first + "".join(digits) + "".join(right_digits), error


def _nearest_pattern(chunk: str, patterns: dict[str, str]) -> tuple[str, int]:
    pattern, digit = min(patterns.items(), key=lambda item: _hamming(chunk, item[0]))
    return digit, _hamming(chunk, pattern)


def _hamming(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b) + abs(len(left) - len(right))


def _ean13_ok(value: str) -> bool:
    if not re.fullmatch(r"\d{13}", value or ""):
        return False
    checksum = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(value[:12]))
    return (10 - checksum % 10) % 10 == int(value[12])


def _duration_ms(started: float) -> int:
    return max(int(round((perf_counter() - started) * 1000)), 0)


def _config(context: PipelineContext) -> _BarcodeBarsConfig:
    raw = context.config.get("barcode_bars_decode", {})
    if not isinstance(raw, dict):
        raw = {}
    return _BarcodeBarsConfig(
        enabled=_bool(raw.get("enabled"), default=True),
        max_tracks=_positive_int(raw.get("max_tracks"), 120),
        min_confidence=_positive_float(raw.get("min_confidence"), 0.70),
    )


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
