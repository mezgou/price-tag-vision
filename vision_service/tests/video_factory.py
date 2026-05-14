from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_synthetic_video(
    destination: Path,
    *,
    width: int = 320,
    height: int = 180,
    fps: float = 5.0,
    frame_count: int = 10,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open synthetic video writer: {destination}")

    try:
        for frame_index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (
                (frame_index * 15) % 255,
                (frame_index * 30) % 255,
                (frame_index * 45) % 255,
            )
            cv2.putText(
                frame,
                f"F{frame_index:02d}",
                (20, min(height - 20, 60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return destination
