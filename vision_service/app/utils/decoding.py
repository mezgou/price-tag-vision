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
    rotation_degrees: float = 0.0


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
    include_resized_x3: bool = False,
    include_resized_x4: bool = False,
    include_adaptive_threshold: bool = True,
    include_right_angle_rotations: bool = False,
    include_small_angle_rotations: bool = False,
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
        variants["resized_x2"] = _resize_variant(crop, name="resized_x2", scale=2)
    if include_resized_x3:
        variants["resized_x3"] = _resize_variant(crop, name="resized_x3", scale=3)
    if include_resized_x4:
        variants["resized_x4"] = _resize_variant(crop, name="resized_x4", scale=4)
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

    if include_right_angle_rotations:
        for angle, image in _right_angle_rotations(crop).items():
            variants[f"rotated_{angle}"] = DecodeVariantImage(
                name=f"rotated_{angle}",
                image=image,
                rotation_degrees=float(angle),
            )
            gray_rotated = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variants[f"rotated_{angle}_clahe"] = DecodeVariantImage(
                name=f"rotated_{angle}_clahe",
                image=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_rotated),
                rotation_degrees=float(angle),
            )
            variants[f"rotated_{angle}_x2"] = _resize_variant(
                image,
                name=f"rotated_{angle}_x2",
                scale=2,
                rotation_degrees=float(angle),
            )

    if include_small_angle_rotations:
        for angle in (-15, -8, 8, 15):
            variants[f"tilt_{angle:+d}"] = DecodeVariantImage(
                name=f"tilt_{angle:+d}",
                image=_rotate_bound(crop, angle),
                rotation_degrees=float(angle),
            )

    return variants


def _adaptive_threshold_block_size(width: int, height: int) -> int:
    base = min(width, height)
    block_size = min(base if base % 2 == 1 else base - 1, 31)
    return max(block_size, 3)


def _resize_variant(
    crop: DecodeImage,
    *,
    name: str,
    scale: int,
    rotation_degrees: float = 0.0,
) -> DecodeVariantImage:
    height, width = crop.shape[:2]
    resized = cv2.resize(
        crop,
        (max(width * scale, 1), max(height * scale, 1)),
        interpolation=cv2.INTER_CUBIC,
    )
    return DecodeVariantImage(
        name=name,
        image=resized,
        scale_x=float(scale),
        scale_y=float(scale),
        rotation_degrees=rotation_degrees,
    )


def _right_angle_rotations(crop: DecodeImage) -> dict[int, DecodeImage]:
    return {
        90: cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(crop, cv2.ROTATE_180),
        270: cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }


def _rotate_bound(crop: DecodeImage, angle_degrees: int) -> DecodeImage:
    height, width = crop.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int((height * sin_value) + (width * cos_value))
    new_height = int((height * cos_value) + (width * sin_value))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(
        crop,
        matrix,
        (max(new_width, 1), max(new_height, 1)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
