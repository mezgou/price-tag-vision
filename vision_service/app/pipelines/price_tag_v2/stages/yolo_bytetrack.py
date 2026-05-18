"""Streaming YOLO + ByteTrack detection/tracking (replaces per-frame detect +
greedy SimpleTracking).

The robot moves, so greedy IoU association across sparsely sampled frames
shatters each price tag into many micro-tracks. Ultralytics' ByteTrack uses a
Kalman motion model and keeps tracker state across calls (``persist=True``),
so feeding the already-undistorted+upright runtime frames *in temporal order*
yields one stable ``track_id`` per physical tag. RowFusion then majority-votes
each field across all of a track's observations into one clean row.
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import cv2

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.yolo_detection import (
    YoloDetectionConfig,
    _nms_detection_candidates,
    _resolve_existing_weights,
    _resolve_yolo_device,
)
from app.schemas.detections import BoundingBox, DetectionCandidate
from app.utils.image_processing import clip_bbox_to_frame

PRICE_TAG_LABEL = "price_tag"
_TRACKER = "bytetrack.yaml"


class YoloByteTrackStage(BaseStage):
    name = "YoloByteTrackStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = YoloDetectionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "sampled_frames_count": len(context.sampled_frames),
            "weights_path": str(config.weights_path),
            "imgsz": config.imgsz,
            "tracker": _TRACKER,
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
        if weights_path is None or find_spec("ultralytics") is None:
            warnings.append(
                "YOLO weights or ultralytics unavailable; "
                "falling back to the heuristic detector."
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

        from ultralytics import YOLO  # type: ignore[import-not-found]

        model = YOLO(str(weights_path))
        device = _resolve_yolo_device(config.device, warnings=warnings)

        detections: list[DetectionCandidate] = []
        frames_processed = 0
        track_ids: set[int] = set()
        synthetic_counter = 0

        # context.sampled_frames are appended in temporal order by
        # FrameSamplingStage, so iterating them feeds ByteTrack a coherent
        # (undistorted + upright) stream.
        for sequence_number, frame_meta in enumerate(context.sampled_frames, start=1):
            if frame_meta.local_frame_path is None or not frame_meta.local_frame_path.exists():
                continue
            frame = cv2.imread(str(frame_meta.local_frame_path))
            if frame is None:
                continue
            frame_height, frame_width = frame.shape[:2]

            try:
                result = model.track(
                    source=frame,
                    imgsz=config.imgsz,
                    conf=config.confidence_threshold,
                    iou=config.iou_threshold,
                    device=device,
                    persist=True,
                    tracker=_TRACKER,
                    verbose=False,
                )[0]
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"ByteTrack failed on frame {frame_meta.frame_index}: {exc}"
                )
                continue
            frames_processed += 1

            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )

            frame_candidates: list[DetectionCandidate] = []
            for rank, (box, conf, tid) in enumerate(zip(xyxy, confs, ids), start=1):
                x_min = max(int(round(float(box[0]))), 0)
                y_min = max(int(round(float(box[1]))), 0)
                x_max = max(int(round(float(box[2]))), x_min + 1)
                y_max = max(int(round(float(box[3]))), y_min + 1)
                clipped = clip_bbox_to_frame(
                    BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                if clipped is None:
                    continue
                if tid is None:
                    synthetic_counter += 1
                    track_key = f"untracked_{synthetic_counter:06d}"
                else:
                    track_ids.add(int(tid))
                    track_key = f"track_{int(tid):05d}"
                frame_candidates.append(
                    DetectionCandidate(
                        detection_id=(
                            f"frame_{frame_meta.frame_index:06d}_bt_"
                            f"{sequence_number:03d}_{rank:02d}"
                        ),
                        frame_index=frame_meta.frame_index,
                        timestamp_ms=frame_meta.timestamp_ms,
                        label=PRICE_TAG_LABEL,
                        bbox=clipped,
                        confidence=round(float(conf), 4),
                        source=config.source,
                        attributes={
                            "rank": rank,
                            "track_id": track_key,
                            "tracking_source": "ultralytics_bytetrack",
                        },
                    )
                )

            detections.extend(
                _nms_detection_candidates(
                    frame_candidates,
                    iou_threshold=config.iou_threshold,
                    max_detections=config.max_detections_per_frame,
                )
            )

        context.detections = detections
        context.artifacts["tracks"] = sorted(
            f"track_{tid:05d}" for tid in track_ids
        )
        return StageOutcome(
            output_summary={
                "enabled": True,
                "weights_loaded": True,
                "weights_path": str(weights_path),
                "frames_processed": frames_processed,
                "detections_count": len(detections),
                "unique_tracks": len(track_ids),
                "device": device,
                "tracker": _TRACKER,
            },
            warnings=warnings,
        )
