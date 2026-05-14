from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.heuristic_candidate_detection import (
    HeuristicCandidateDetectionStage,
)


def test_heuristic_candidate_detection_finds_rectangular_candidates(
    pipeline_context,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (70, 180), (260, 245), (245, 245, 245), thickness=-1)
    cv2.rectangle(frame, (220, 188), (250, 215), (0, 190, 255), thickness=-1)
    cv2.rectangle(frame, (320, 220), (520, 290), (250, 250, 250), thickness=-1)
    cv2.rectangle(frame, (470, 228), (510, 258), (0, 220, 255), thickness=-1)

    frame_path = tmp_path / "runtime" / "frame_000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["candidate_detection"] = {
        "enabled": True,
        "source": "heuristic_color_geometry_v1",
        "max_candidates_per_frame": 10,
        "min_area_ratio": 0.001,
        "max_area_ratio": 0.2,
        "min_aspect_ratio": 1.2,
        "max_aspect_ratio": 8.0,
        "nms_iou_threshold": 0.4,
        "debug_save_overlays": True,
        "debug_overlay_jpeg_quality": 85,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=5,
            timestamp_ms=1000,
            original_width=640,
            original_height=360,
            processed_width=640,
            processed_height=360,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.frames_processed = 1

    outcome = HeuristicCandidateDetectionStage().run(pipeline_context)

    assert outcome.output_summary["frames_processed"] == 1
    assert outcome.output_summary["detections_count"] >= 2
    assert len(pipeline_context.detections) >= 2
    assert all(
        detection.label == "price_tag_candidate"
        for detection in pipeline_context.detections
    )
    assert all(
        detection.source == "heuristic_color_geometry_v1"
        for detection in pipeline_context.detections
    )
    first_bbox = pipeline_context.detections[0].bbox
    assert first_bbox.x_min < first_bbox.x_max
    assert first_bbox.y_min < first_bbox.y_max
    assert len(pipeline_context.debug_overlay_keys) == 1
    assert pipeline_context.artifacts["debug_overlay_keys"] == [
        "outputs/job-123/debug/overlays/frame_000001_detections.jpg"
    ]
    assert "outputs/job-123/debug/overlays/frame_000001_detections.jpg" in in_memory_storage.objects


def test_heuristic_candidate_detection_handles_dark_frame_without_failure(
    pipeline_context,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_path = tmp_path / "runtime" / "frame_000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["candidate_detection"] = {
        "enabled": True,
        "debug_save_overlays": True,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=0,
            timestamp_ms=0,
            original_width=320,
            original_height=240,
            processed_width=320,
            processed_height=240,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.frames_processed = 1

    outcome = HeuristicCandidateDetectionStage().run(pipeline_context)

    assert outcome.output_summary["detections_count"] == 0
    assert pipeline_context.detections == []
    assert len(pipeline_context.debug_overlay_keys) == 1
    assert "outputs/job-123/debug/overlays/frame_000001_detections.jpg" in in_memory_storage.objects
