from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from app.pipelines.base import PipelineContext, SampledFrameMetadata, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import (
    FrameSamplingConfig,
    FrameSamplingStage,
    _positive_capture_float,
    _resolve_sample_interval,
    build_camera_model,
)


@dataclass(slots=True)
class _MotionAwareConfig:
    enabled: bool
    dense_sample_fps: float
    sparse_sample_fps: float
    low_motion_threshold: float
    high_motion_threshold: float


class V3FrameSamplingStage(FrameSamplingStage):
    name = "V3FrameSamplingStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        summary = super().describe_input(context)
        motion = _motion_config(context)
        summary["motion_aware"] = {
            "enabled": motion.enabled,
            "dense_sample_fps": motion.dense_sample_fps,
            "sparse_sample_fps": motion.sparse_sample_fps,
            "low_motion_threshold": motion.low_motion_threshold,
            "high_motion_threshold": motion.high_motion_threshold,
        }
        return summary

    def run(self, context: PipelineContext) -> StageOutcome:
        motion = _motion_config(context)
        if not motion.enabled:
            return super().run(context)

        if not context.local_video_path.exists():
            raise RuntimeError(
                f"Local video file does not exist: {context.local_video_path}"
            )

        config = FrameSamplingConfig.from_context(context)
        warnings = list(config.warnings)
        camera = build_camera_model(context)

        capture = cv2.VideoCapture(str(context.local_video_path))
        if not capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video file: {context.local_video_path}"
            )

        sampled_frames: list[SampledFrameMetadata] = []
        debug_frame_keys: list[str] = []
        motion_values: list[float] = []
        dense_selected = 0
        sparse_selected = 0
        previous_gray = None

        try:
            source_fps = context.video_metadata.fps or _positive_capture_float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            dense_interval = _resolve_sample_interval(
                source_fps=source_fps,
                sample_fps=motion.dense_sample_fps,
            )
            sparse_interval = _resolve_sample_interval(
                source_fps=source_fps,
                sample_fps=motion.sparse_sample_fps,
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

                gray = _flow_gray(frame)
                motion_score = _motion_score(previous_gray, gray)
                previous_gray = gray
                if motion_score is not None:
                    motion_values.append(motion_score)

                is_sparse_tick = frame_index % sparse_interval == 0
                is_dense_tick = frame_index % dense_interval == 0
                low_motion = (
                    motion_score is None
                    or motion_score <= motion.low_motion_threshold
                    or motion_score <= motion.high_motion_threshold
                )
                should_sample = is_sparse_tick or (is_dense_tick and low_motion)

                if not should_sample:
                    frame_index += 1
                    continue

                sampled_frame, debug_frame_key = self._process_frame(
                    context=context,
                    frame=frame,
                    frame_index=frame_index,
                    source_fps=source_fps,
                    sequence_number=len(sampled_frames) + 1,
                    config=config,
                    camera=camera,
                )
                sampled_frames.append(sampled_frame)
                if is_sparse_tick:
                    sparse_selected += 1
                elif is_dense_tick:
                    dense_selected += 1
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
        context.artifacts["motion_sampling"] = {
            "enabled": True,
            "dense_selected": dense_selected,
            "sparse_selected": sparse_selected,
            "motion_mean": _mean(motion_values),
            "motion_min": min(motion_values) if motion_values else 0.0,
            "motion_max": max(motion_values) if motion_values else 0.0,
        }

        return StageOutcome(
            output_summary={
                "frames_processed": context.frames_processed,
                "sampled_frames_count": len(sampled_frames),
                "debug_frames_saved": len(debug_frame_keys),
                "orientation_mode": config.orientation_mode,
                "motion_aware": context.artifacts["motion_sampling"],
            },
            warnings=warnings,
        )


def _motion_config(context: PipelineContext) -> _MotionAwareConfig:
    raw = context.config.get("frame_sampling", {})
    motion_raw = raw.get("motion_aware", {}) if isinstance(raw, dict) else {}
    if not isinstance(motion_raw, dict):
        motion_raw = {}
    sample_fps = _positive_float(raw.get("sample_fps"), 5.0) if isinstance(raw, dict) else 5.0
    return _MotionAwareConfig(
        enabled=_bool(motion_raw.get("enabled"), default=False),
        dense_sample_fps=_positive_float(motion_raw.get("dense_sample_fps"), sample_fps),
        sparse_sample_fps=_positive_float(motion_raw.get("sparse_sample_fps"), max(sample_fps / 3, 1.0)),
        low_motion_threshold=_positive_float(motion_raw.get("low_motion_threshold"), 7.5),
        high_motion_threshold=_positive_float(motion_raw.get("high_motion_threshold"), 24.0),
    )


def _flow_gray(frame: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(320.0 / max(width, 1), 180.0 / max(height, 1), 1.0)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def _motion_score(previous_gray: Any, gray: Any) -> float | None:
    if previous_gray is None or previous_gray.shape != gray.shape:
        return None
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        0.5,
        2,
        15,
        2,
        5,
        1.2,
        0,
    )
    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
