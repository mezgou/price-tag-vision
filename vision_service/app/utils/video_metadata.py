from __future__ import annotations

from pathlib import Path

from app.pipelines.base import VideoMetadata


def read_video_metadata(
    video_path: Path,
    *,
    input_video_key: str,
) -> tuple[VideoMetadata, list[str]]:
    metadata = VideoMetadata.from_input_key(input_video_key)
    warnings: list[str] = []

    try:
        import cv2
    except ImportError:
        warnings.append("OpenCV is not available; video metadata extraction was skipped.")
        return metadata, warnings

    if not video_path.exists():
        warnings.append(f"Video file does not exist locally: {video_path}")
        return metadata, warnings

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        warnings.append(f"OpenCV could not open video file: {video_path}")
        return metadata, warnings

    try:
        fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _positive_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata.fps = fps
        metadata.frame_count = frame_count
        metadata.width = width
        metadata.height = height

        if fps is not None and frame_count is not None:
            metadata.duration_ms = int(round((frame_count / fps) * 1000))
    finally:
        capture.release()

    return metadata, warnings


def _positive_float(value: float) -> float | None:
    if value <= 0:
        return None
    return round(float(value), 3)


def _positive_int(value: float) -> int | None:
    if value <= 0:
        return None
    return int(round(value))
