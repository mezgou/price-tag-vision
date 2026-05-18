"""Quick visual check: which orientation mode makes price tags upright.

Reads the doc reference frame (26_12-20 @ 6595ms, bbox 2011,1923->2231,2115)
and renders all 4 orientation variants with the GT bbox transformed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vision_service"))
sys.path.insert(1, str(REPO))

from app.datasets.yolo_price_tag_dataset import FloatBoundingBox, rotate_bbox
from app.pipelines.price_tag_cpu_v1.orientation import (
    SUPPORTED_ORIENTATION_MODES,
    apply_orientation,
)

VIDEO = REPO / "data/videos/26_12-20/26_12-20.mp4"
TS_MS = 6595
BBOX = FloatBoundingBox(x_min=2011.9, y_min=1923.3, x_max=2231.6, y_max=2115.3)
OUT = REPO / "artifacts/orient_check"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(round(TS_MS * fps / 1000.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("FAILED to read frame")
        return 1
    h, w = frame.shape[:2]
    print(f"source {w}x{h} fps={fps} frame_idx={frame_idx}")
    for mode in SUPPORTED_ORIENTATION_MODES:
        img = apply_orientation(frame, mode).copy()
        rb = rotate_bbox(BBOX, mode=mode, source_width=w, source_height=h)
        p1 = (int(rb.x_min), int(rb.y_min))
        p2 = (int(rb.x_max), int(rb.y_max))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 6)
        # downscale so it's viewable
        scale = 900.0 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale)
        path = OUT / f"{mode}.jpg"
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{mode}: img={img.shape[1]}x{img.shape[0]} bbox=({p1},{p2}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
