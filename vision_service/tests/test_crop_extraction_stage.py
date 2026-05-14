from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.crop_extraction import CropExtractionStage
from app.schemas.detections import BoundingBox, DetectionCandidate


def test_crop_extraction_stage_saves_crops_and_quality_metadata(
    pipeline_context,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 120), (240, 170), (245, 245, 245), thickness=-1)
    cv2.putText(
        frame,
        "PRICE",
        (72, 152),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    frame_path = tmp_path / "runtime" / "frame_000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["crop_extraction"] = {
        "enabled": True,
        "source": "crop_extraction_v1",
        "bbox_padding_ratio": 0.08,
        "min_crop_width": 24,
        "min_crop_height": 16,
        "max_crops_per_frame": 10,
        "max_total_crops": 20,
        "debug_save_crops": True,
        "debug_crop_jpeg_quality": 90,
        "top_crops_preview_limit": 20,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=5,
            timestamp_ms=1000,
            original_width=320,
            original_height=240,
            processed_width=320,
            processed_height=240,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.detections = [
        DetectionCandidate(
            detection_id="frame_000005_candidate_001_01",
            frame_index=5,
            timestamp_ms=1000,
            label="price_tag_candidate",
            bbox=BoundingBox(x_min=60, y_min=120, x_max=240, y_max=170),
            confidence=0.91,
            source="heuristic_color_geometry_v1",
        )
    ]

    outcome = CropExtractionStage().run(pipeline_context)

    assert outcome.output_summary["crops_count"] == 1
    assert outcome.output_summary["debug_crops_count"] == 1
    assert len(pipeline_context.crop_candidates) == 1
    assert len(pipeline_context.debug_crop_keys) == 1
    crop = pipeline_context.crop_candidates[0]
    assert crop.source == "crop_extraction_v1"
    assert crop.crop_key == "outputs/job-123/debug/crops/frame_000001_det_000001.jpg"
    assert crop.width >= 24
    assert crop.height >= 16
    assert crop.quality.score >= 0
    assert crop.padded_bbox.x_min <= crop.bbox.x_min
    assert crop.padded_bbox.y_min <= crop.bbox.y_min
    assert crop.crop_key in in_memory_storage.objects


def test_crop_extraction_stage_handles_empty_detections(
    pipeline_context,
) -> None:
    pipeline_context.config["crop_extraction"] = {"enabled": True}
    pipeline_context.detections = []

    outcome = CropExtractionStage().run(pipeline_context)

    assert outcome.output_summary["crops_count"] == 0
    assert pipeline_context.crop_candidates == []
    assert pipeline_context.debug_crop_keys == []


def test_crop_extraction_stage_clips_out_of_bounds_bbox_without_failure(
    pipeline_context,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 40), (120, 90), (255, 255, 255), thickness=-1)
    frame_path = tmp_path / "runtime" / "frame_000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["crop_extraction"] = {
        "enabled": True,
        "min_crop_width": 20,
        "min_crop_height": 10,
        "debug_save_crops": True,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=0,
            timestamp_ms=0,
            original_width=200,
            original_height=100,
            processed_width=200,
            processed_height=100,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.detections = [
        DetectionCandidate(
            detection_id="frame_000000_candidate_001_01",
            frame_index=0,
            timestamp_ms=0,
            label="price_tag_candidate",
            bbox=BoundingBox(x_min=0, y_min=35, x_max=260, y_max=110),
            confidence=0.88,
            source="heuristic_color_geometry_v1",
        )
    ]

    outcome = CropExtractionStage().run(pipeline_context)

    assert outcome.output_summary["crops_count"] == 1
    crop = pipeline_context.crop_candidates[0]
    assert crop.bbox.x_max == 200
    assert crop.bbox.y_max == 100
    assert crop.crop_key in in_memory_storage.objects
