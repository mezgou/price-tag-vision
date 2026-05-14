from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

DecodeImage = NDArray[np.uint8]


@dataclass(slots=True)
class DecodeVariantImage:
    name: str
    image: DecodeImage
    scale_x: float = 1.0
    scale_y: float = 1.0


def normalize_decoded_payload(payload: str | bytes | None) -> str:
    if payload is None:
        return ""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="ignore")

    return payload.replace("\x00", "").strip()


def build_payload_preview(payload: str, *, max_length: int) -> str:
    if max_length <= 0:
        return ""

    normalized = normalize_decoded_payload(payload)
    if len(normalized) <= max_length:
        return normalized

    if max_length <= 3:
        return normalized[:max_length]
    return normalized[: max_length - 3] + "..."


def build_decode_variants(
    crop: DecodeImage,
    *,
    include_original: bool = True,
    include_grayscale: bool = True,
    include_clahe: bool = True,
    include_sharpened: bool = True,
    include_resized_x2: bool = True,
    include_adaptive_threshold: bool = True,
) -> dict[str, DecodeVariantImage]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants: dict[str, DecodeVariantImage] = {}

    if include_original:
        variants["original"] = DecodeVariantImage(name="original", image=crop)
    if include_grayscale:
        variants["grayscale"] = DecodeVariantImage(name="grayscale", image=gray)
    if include_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants["clahe"] = DecodeVariantImage(name="clahe", image=clahe)
    if include_sharpened:
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        variants["sharpened"] = DecodeVariantImage(name="sharpened", image=sharpened)
    if include_resized_x2:
        height, width = crop.shape[:2]
        resized_x2 = cv2.resize(
            crop,
            (max(width * 2, 1), max(height * 2, 1)),
            interpolation=cv2.INTER_CUBIC,
        )
        variants["resized_x2"] = DecodeVariantImage(
            name="resized_x2",
            image=resized_x2,
            scale_x=2.0,
            scale_y=2.0,
        )
    if include_adaptive_threshold:
        block_size = _adaptive_threshold_block_size(gray.shape[1], gray.shape[0])
        adaptive_threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            7,
        )
        variants["adaptive_threshold"] = DecodeVariantImage(
            name="adaptive_threshold",
            image=adaptive_threshold,
        )

    return variants


def _adaptive_threshold_block_size(width: int, height: int) -> int:
    base = min(width, height)
    block_size = min(base if base % 2 == 1 else base - 1, 31)
    return max(block_size, 3)
