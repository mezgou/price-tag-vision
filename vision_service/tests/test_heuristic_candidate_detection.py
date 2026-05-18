from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.heuristic_candidate_detection import (
    CandidateDetectionConfig,
    CandidateFeatures,
    HeuristicCandidateDetectionStage,
    score_candidate_features,
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
        "source": "heuristic_color_geometry_v2",
        "max_candidates_per_frame": 10,
        "min_area_ratio": 0.001,
        "max_area_ratio": 0.2,
        "min_aspect_ratio": 1.2,
        "max_aspect_ratio": 8.0,
        "nms_iou_threshold": 0.4,
        "debug_save_overlays": True,
        "debug_save_masks": True,
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
        detection.source == "heuristic_color_geometry_v2"
        for detection in pipeline_context.detections
    )
    first_attributes = pipeline_context.detections[0].attributes
    assert "area_ratio" in first_attributes
    assert "aspect_ratio" in first_attributes
    assert "rectangularity" in first_attributes
    assert "white_ratio" in first_attributes
    assert "yellow_or_orange_ratio" in first_attributes
    assert "edge_density" in first_attributes
    assert "text_like_score" in first_attributes
    assert "center_y_ratio" in first_attributes
    assert isinstance(first_attributes["score_components"], dict)
    first_bbox = pipeline_context.detections[0].bbox
    assert first_bbox.x_min < first_bbox.x_max
    assert first_bbox.y_min < first_bbox.y_max
    assert len(pipeline_context.debug_overlay_keys) == 1
    assert len(pipeline_context.debug_mask_keys) == 3
    assert pipeline_context.artifacts["debug_overlay_keys"] == [
        "outputs/job-123/debug/overlays/frame_000001_detections.jpg"
    ]
    assert pipeline_context.artifacts["debug_mask_keys"] == [
        "outputs/job-123/debug/masks/frame_000001_white_mask.jpg",
        "outputs/job-123/debug/masks/frame_000001_accent_mask.jpg",
        "outputs/job-123/debug/masks/frame_000001_combined_mask.jpg",
    ]
    assert "outputs/job-123/debug/overlays/frame_000001_detections.jpg" in in_memory_storage.objects
    assert "outputs/job-123/debug/masks/frame_000001_white_mask.jpg" in in_memory_storage.objects
    assert "outputs/job-123/debug/masks/frame_000001_accent_mask.jpg" in in_memory_storage.objects
    assert "outputs/job-123/debug/masks/frame_000001_combined_mask.jpg" in in_memory_storage.objects


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
        "debug_save_masks": False,
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
    assert pipeline_context.debug_mask_keys == []
    assert "outputs/job-123/debug/overlays/frame_000001_detections.jpg" in in_memory_storage.objects


def test_candidate_scoring_helper_prefers_rectangular_light_object_over_noise() -> None:
    config = CandidateDetectionConfig(
        enabled=True,
        source="heuristic_color_geometry_v2",
        max_candidates_per_frame=10,
        min_area_ratio=0.00008,
        max_area_ratio=0.12,
        min_aspect_ratio=1.1,
        max_aspect_ratio=9.5,
        nms_iou_threshold=0.4,
        debug_save_overlays=True,
        debug_overlay_jpeg_quality=85,
        debug_save_masks=True,
        debug_save_rejected=False,
        top_rejected_preview_limit=20,
        warnings=[],
    )
    strong_candidate = CandidateFeatures(
        area_ratio=0.008,
        aspect_ratio=2.7,
        rectangularity=0.91,
        white_ratio=0.64,
        yellow_or_orange_ratio=0.07,
        edge_density=0.11,
        text_like_score=0.56,
        center_y_ratio=0.72,
        glare_ratio=0.04,
    )
    noisy_candidate = CandidateFeatures(
        area_ratio=0.03,
        aspect_ratio=1.0,
        rectangularity=0.38,
        white_ratio=0.08,
        yellow_or_orange_ratio=0.0,
        edge_density=0.01,
        text_like_score=0.03,
        center_y_ratio=0.08,
        glare_ratio=0.62,
    )

    strong_score, _ = score_candidate_features(features=strong_candidate, config=config)
    noisy_score, _ = score_candidate_features(features=noisy_candidate, config=config)

    assert strong_score > noisy_score
    assert strong_score > 0.5
