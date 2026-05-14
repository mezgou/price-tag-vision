from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipelines.base import (
    BaseStage,
    PipelineContext,
    SampledFrameMetadata,
    StageOutcome,
)
from app.pipelines.price_tag_cpu_v1.orientation import (
    OrientationMode,
    apply_orientation,
    resolve_orientation_mode,
)
from app.utils.image_processing import resize_to_max_width

FrameArray = NDArray[np.uint8]
DEFAULT_DEBUG_MAX_WIDTH = 1280


@dataclass(slots=True)
class FrameSamplingConfig:
    sample_fps: float
    max_frames: int
    orientation_mode: OrientationMode
    debug_save_frames: bool
    debug_jpeg_quality: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "FrameSamplingConfig":
        raw_config = context.config.get("frame_sampling", {})
        warnings: list[str] = []

        if not isinstance(raw_config, dict):
            warnings.append(
                "Frame sampling config must be a mapping. Falling back to defaults."
            )
            raw_config = {}

        sample_fps = _coerce_positive_float(
            raw_config.get("sample_fps"),
            default=1.0,
            field_name="sample_fps",
            warnings=warnings,
        )
        max_frames = _coerce_positive_int(
            raw_config.get("max_frames"),
            default=20,
            field_name="max_frames",
            warnings=warnings,
        )
        orientation_mode, orientation_warnings = resolve_orientation_mode(
            raw_config.get("orientation_mode")
        )
        warnings.extend(orientation_warnings)
        debug_save_frames = _coerce_bool(
            raw_config.get("debug_save_frames"),
            default=True,
            field_name="debug_save_frames",
            warnings=warnings,
        )
        debug_jpeg_quality = _coerce_int_in_range(
            raw_config.get("debug_jpeg_quality"),
            default=85,
            min_value=1,
            max_value=100,
            field_name="debug_jpeg_quality",
            warnings=warnings,
        )

        return cls(
            sample_fps=sample_fps,
            max_frames=max_frames,
            orientation_mode=orientation_mode,
            debug_save_frames=debug_save_frames,
            debug_jpeg_quality=debug_jpeg_quality,
            warnings=warnings,
        )


class FrameSamplingStage(BaseStage):
    name = "FrameSamplingStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = FrameSamplingConfig.from_context(context)
        return {
            "local_video_path": str(context.local_video_path),
            "metadata_fps": context.video_metadata.fps,
            "metadata_frame_count": context.video_metadata.frame_count,
            "sample_fps": config.sample_fps,
            "max_frames": config.max_frames,
            "orientation_mode": config.orientation_mode,
            "debug_save_frames": config.debug_save_frames,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        if not context.local_video_path.exists():
            raise RuntimeError(
                f"Local video file does not exist: {context.local_video_path}"
            )

        config = FrameSamplingConfig.from_context(context)
        warnings = list(config.warnings)

        capture = cv2.VideoCapture(str(context.local_video_path))
        if not capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video file: {context.local_video_path}"
            )

        sampled_frames: list[SampledFrameMetadata] = []
        debug_frame_keys: list[str] = []

        try:
            source_fps = context.video_metadata.fps or _positive_capture_float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            sample_interval = _resolve_sample_interval(
                source_fps=source_fps,
                sample_fps=config.sample_fps,
            )

            frame_index = 0
            while len(sampled_frames) < config.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame is None:
                    warnings.append(
                        f"Frame {frame_index} could not be decoded and was skipped."
                    )
                    frame_index += 1
                    continue

                if frame_index % sample_interval != 0:
                    frame_index += 1
                    continue

                sampled_frame, debug_frame_key = self._process_frame(
                    context=context,
                    frame=frame,
                    frame_index=frame_index,
                    source_fps=source_fps,
                    sequence_number=len(sampled_frames) + 1,
                    config=config,
                )
                sampled_frames.append(sampled_frame)
                if debug_frame_key is not None:
                    debug_frame_keys.append(debug_frame_key)

                frame_index += 1

            if not sampled_frames:
                warnings.append("No frames were sampled from the video.")
        finally:
            capture.release()

        context.sampled_frames = sampled_frames
        context.frames_processed = len(sampled_frames)
        context.debug_frame_keys = debug_frame_keys
        context.artifacts["debug_frame_keys"] = list(debug_frame_keys)

        return StageOutcome(
            output_summary={
                "frames_processed": context.frames_processed,
                "sampled_frames_count": len(sampled_frames),
                "debug_frames_saved": len(debug_frame_keys),
                "orientation_mode": config.orientation_mode,
            },
            warnings=warnings,
        )

    def _process_frame(
        self,
        *,
        context: PipelineContext,
        frame: FrameArray,
        frame_index: int,
        source_fps: float | None,
        sequence_number: int,
        config: FrameSamplingConfig,
    ) -> tuple[SampledFrameMetadata, str | None]:
        original_height, original_width = frame.shape[:2]
        oriented_frame = apply_orientation(frame, config.orientation_mode)
        processed_frame = resize_to_max_width(
            oriented_frame,
            max_width=DEFAULT_DEBUG_MAX_WIDTH,
        )
        processed_height, processed_width = processed_frame.shape[:2]

        local_frame_path = _write_runtime_frame(
            context=context,
            frame=processed_frame,
            sequence_number=sequence_number,
            jpeg_quality=config.debug_jpeg_quality,
        )
        debug_frame_key: str | None = None
        if config.debug_save_frames:
            debug_frame_key = _upload_debug_frame(
                context=context,
                frame=processed_frame,
                sequence_number=sequence_number,
                jpeg_quality=config.debug_jpeg_quality,
            )

        metadata = SampledFrameMetadata(
            frame_index=frame_index,
            timestamp_ms=_frame_timestamp_ms(frame_index=frame_index, source_fps=source_fps),
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            orientation_applied=config.orientation_mode,
            debug_frame_key=debug_frame_key,
            local_frame_path=local_frame_path,
        )
        return metadata, debug_frame_key


def _upload_debug_frame(
    *,
    context: PipelineContext,
    frame: FrameArray,
    sequence_number: int,
    jpeg_quality: int,
) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError(
            f"Failed to encode sampled frame {sequence_number} as JPEG."
        )

    relative_path = f"debug/frames/frame_{sequence_number:06d}.jpg"
    return context.artifact_writer.upload_bytes(
        relative_path,
        encoded.tobytes(),
        "image/jpeg",
    )


def _write_runtime_frame(
    *,
    context: PipelineContext,
    frame: FrameArray,
    sequence_number: int,
    jpeg_quality: int,
) -> Path:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise RuntimeError(
            f"Failed to encode sampled frame {sequence_number} for runtime storage."
        )

    runtime_dir = context.work_dir / "runtime" / "sampled_frames"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / f"frame_{sequence_number:06d}.jpg"
    destination.write_bytes(encoded.tobytes())
    return destination


def _resolve_sample_interval(*, source_fps: float | None, sample_fps: float) -> int:
    if source_fps is None or source_fps <= 0:
        return 1
    return max(int(round(source_fps / sample_fps)), 1)


def _frame_timestamp_ms(*, frame_index: int, source_fps: float | None) -> int | None:
    if source_fps is None or source_fps <= 0:
        return None
    return int(round((frame_index / source_fps) * 1000))


def _positive_capture_float(value: float) -> float | None:
    if value <= 0:
        return None
    return float(value)


def _coerce_positive_float(
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
        warnings.append(
            f"Invalid {field_name} '{value}'. Falling back to {default}."
        )
        return default

    if coerced <= 0:
        warnings.append(
            f"{field_name} must be > 0. Falling back to {default}."
        )
        return default
    return coerced


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
        warnings.append(
            f"Invalid {field_name} '{value}'. Falling back to {default}."
        )
        return default

    if coerced <= 0:
        warnings.append(
            f"{field_name} must be > 0. Falling back to {default}."
        )
        return default
    return coerced


def _coerce_int_in_range(
    value: Any,
    *,
    default: int,
    min_value: int,
    max_value: int,
    field_name: str,
    warnings: list[str],
) -> int:
    if value is None:
        return default

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        warnings.append(
            f"Invalid {field_name} '{value}'. Falling back to {default}."
        )
        return default

    if coerced < min_value or coerced > max_value:
        warnings.append(
            f"{field_name} must be between {min_value} and {max_value}. Falling back to {default}."
        )
        return default
    return coerced


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

    warnings.append(
        f"Invalid {field_name} '{value}'. Falling back to {default}."
    )
    return default
