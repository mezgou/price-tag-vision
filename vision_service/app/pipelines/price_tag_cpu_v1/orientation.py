from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

OrientationMode = Literal[
    "none",
    "rotate_90_cw",
    "rotate_90_ccw",
    "rotate_180",
]

SUPPORTED_ORIENTATION_MODES: tuple[OrientationMode, ...] = (
    "none",
    "rotate_90_cw",
    "rotate_90_ccw",
    "rotate_180",
)

FrameArray = NDArray[np.uint8]


def resolve_orientation_mode(raw_value: Any) -> tuple[OrientationMode, list[str]]:
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
    else:
        candidate = ""

    if candidate in SUPPORTED_ORIENTATION_MODES:
        return candidate, []

    if raw_value in (None, ""):
        return "rotate_90_ccw", []

    return (
        "none",
        [f"Unknown orientation_mode '{raw_value}'. Falling back to 'none'."],
    )


def apply_orientation(frame: FrameArray, mode: OrientationMode) -> FrameArray:
    if mode == "none":
        return frame
    if mode == "rotate_90_cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "rotate_90_ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "rotate_180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame
