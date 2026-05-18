from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import cv2

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import BoundingBox, DetectionCandidate
from app.utils.image_processing import clip_bbox_to_frame, non_max_suppress_boxes, ScoredBoundingBox

PRICE_TAG_LABEL = "price_tag"


@dataclass(slots=True)
class YoloDetectionConfig:
    enabled: bool
    weights_path: Path
    fallback_weights_path: Path | None
    source: str
    confidence_threshold: float
    iou_threshold: float
    imgsz: int
    device: str
    max_detections_per_frame: int
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "YoloDetectionConfig":
        raw_config = context.config.get("yolo_detection", {})
        warnings: list[str] = []
        if not isinstance(raw_config, dict):
            warnings.append("YOLO detection config must be a mapping. Falling back to defaults.")
            raw_config = {}

        weights_path = _resolve_path(
            raw_config.get("weights_path"),
            default="artifacts/models/price_tag_yolo11n_v1/weights/best.pt",
        )
        fallback_raw = raw_config.get("fallback_weights_path")
        fallback_weights_path = None if fallback_raw in {None, ""} else _resolve_path(fallback_raw)

        return cls(
            enabled=_coerce_bool(raw_config.get("enabled"), default=True, field_name="enabled", warnings=warnings),
            weights_path=weights_path,
            fallback_weights_path=fallback_weights_path,
            source=_coerce_string(raw_config.get("source"), default="yolo11n_price_tag_v2", field_name="source", warnings=warnings),
            confidence_threshold=_coerce_ratio(raw_config.get("confidence_threshold"), default=0.25, field_name="confidence_threshold", warnings=warnings),
            iou_threshold=_coerce_ratio(raw_config.get("iou_threshold"), default=0.50, field_name="iou_threshold", warnings=warnings),
            imgsz=_coerce_positive_int(raw_config.get("imgsz"), default=1280, field_name="imgsz", warnings=warnings),
            device=_coerce_string(raw_config.get("device"), default="auto", field_name="device", warnings=warnings),
            max_detections_per_frame=_coerce_positive_int(
                raw_config.get("max_detections_per_frame"),
                default=32,
                field_name="max_detections_per_frame",
                warnings=warnings,
            ),
            warnings=warnings,
        )


class YoloDetectionStage(BaseStage):
    name = "YoloDetectionStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = YoloDetectionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "sampled_frames_count": len(context.sampled_frames),
            "weights_path": str(config.weights_path),
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "imgsz": config.imgsz,
            "device": config.device,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = YoloDetectionConfig.from_context(context)
        warnings = list(config.warnings)

        if not config.enabled:
            context.detections = []
            return StageOutcome(
                output_summary={"enabled": False, "detections_count": 0},
                warnings=warnings,
            )

        weights_path = _resolve_existing_weights(config)
        if weights_path is None:
            warnings.append(
                "YOLO weights were not found. The v2 pipeline will use the heuristic fallback detector."
            )
            context.detections = []
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "weights_loaded": False,
                    "detections_count": 0,
                },
                warnings=warnings,
            )

        if find_spec("ultralytics") is None:
            warnings.append(
                "Ultralytics is not installed. The v2 pipeline will use the heuristic fallback detector."
            )
            context.detections = []
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "weights_loaded": False,
                    "weights_path": str(weights_path),
                    "detections_count": 0,
                },
                warnings=warnings,
            )

        from ultralytics import YOLO  # type: ignore[import-not-found]

        model = YOLO(str(weights_path))
        detections: list[DetectionCandidate] = []
        frames_processed = 0
        device = _resolve_yolo_device(config.device, warnings=warnings)

        for sequence_number, frame_meta in enumerate(context.sampled_frames, start=1):
            if frame_meta.local_frame_path is None or not frame_meta.local_frame_path.exists():
                warnings.append(f"Sampled frame {frame_meta.frame_index} has no local image.")
                continue

            frame = cv2.imread(str(frame_meta.local_frame_path))
            if frame is None:
                warnings.append(f"Sampled frame {frame_meta.frame_index} could not be read.")
                continue
            frame_height, frame_width = frame.shape[:2]

            prediction = model.predict(
                source=frame,
                imgsz=config.imgsz,
                conf=config.confidence_threshold,
                iou=config.iou_threshold,
                device=device,
                verbose=False,
                save=False,
            )[0]
            frames_processed += 1

            frame_candidates = _prediction_to_candidates(
                prediction=prediction,
                frame_index=frame_meta.frame_index,
                timestamp_ms=frame_meta.timestamp_ms,
                frame_width=frame_width,
                frame_height=frame_height,
                sequence_number=sequence_number,
                source=config.source,
            )
            frame_candidates = _nms_detection_candidates(
                frame_candidates,
                iou_threshold=config.iou_threshold,
                max_detections=config.max_detections_per_frame,
            )
            detections.extend(frame_candidates)

        context.detections = detections
        return StageOutcome(
            output_summary={
                "enabled": True,
                "weights_loaded": True,
                "weights_path": str(weights_path),
                "frames_processed": frames_processed,
                "detections_count": len(detections),
                "source": config.source,
                "device": device,
            },
            warnings=warnings,
        )


def _prediction_to_candidates(
    *,
    prediction: Any,
    frame_index: int,
    timestamp_ms: int | None,
    frame_width: int,
    frame_height: int,
    sequence_number: int,
    source: str,
) -> list[DetectionCandidate]:
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    candidates: list[DetectionCandidate] = []

    for rank, (xyxy, confidence) in enumerate(zip(xyxy_values, confidences), start=1):
        x_min = max(int(round(float(xyxy[0]))), 0)
        y_min = max(int(round(float(xyxy[1]))), 0)
        x_max = max(int(round(float(xyxy[2]))), x_min + 1)
        y_max = max(int(round(float(xyxy[3]))), y_min + 1)
        raw_box = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        clipped = clip_bbox_to_frame(
            raw_box,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if clipped is None:
            continue
        candidates.append(
            DetectionCandidate(
                detection_id=f"frame_{frame_index:06d}_yolo_{sequence_number:03d}_{rank:02d}",
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label=PRICE_TAG_LABEL,
                bbox=clipped,
                confidence=round(float(confidence), 4),
                source=source,
                attributes={"rank": rank},
            )
        )
    return candidates


def _nms_detection_candidates(
    candidates: list[DetectionCandidate],
    *,
    iou_threshold: float,
    max_detections: int,
) -> list[DetectionCandidate]:
    scored = [
        ScoredBoundingBox(
            bbox=candidate.bbox,
            score=candidate.confidence,
            attributes={"candidate": candidate},
        )
        for candidate in candidates
    ]
    kept = non_max_suppress_boxes(
        scored,
        iou_threshold=iou_threshold,
        max_candidates=max_detections,
    )
    return [item.attributes["candidate"] for item in kept]


def _resolve_existing_weights(config: YoloDetectionConfig) -> Path | None:
    if config.weights_path.exists():
        return config.weights_path
    if config.fallback_weights_path is not None and config.fallback_weights_path.exists():
        return config.fallback_weights_path
    return None


def _resolve_yolo_device(requested_device: str, *, warnings: list[str]) -> str:
    normalized = requested_device.strip().lower()
    if normalized not in {"", "auto"}:
        return requested_device

    try:
        import torch
    except ImportError:
        warnings.append("torch is not installed; YOLO inference will use CPU.")
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    warnings.append("torch is installed but CUDA is unavailable; YOLO inference will use CPU.")
    return "cpu"


def _resolve_path(value: Any, *, default: str | None = None) -> Path:
    raw = default if value in {None, ""} else str(value)
    assert raw is not None
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _coerce_bool(value: Any, *, default: bool, field_name: str, warnings: list[str]) -> bool:
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


def _coerce_string(value: Any, *, default: str, field_name: str, warnings: list[str]) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None:
        return default
    warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
    return default


def _coerce_positive_int(value: Any, *, default: int, field_name: str, warnings: list[str]) -> int:
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


def _coerce_ratio(value: Any, *, default: float, field_name: str, warnings: list[str]) -> float:
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0 or coerced > 1:
        warnings.append(f"{field_name} must be within (0, 1]. Falling back to {default}.")
        return default
    return coerced
