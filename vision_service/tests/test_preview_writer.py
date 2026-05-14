from __future__ import annotations

import json

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.preview_writer import PreviewWriterStage
from app.schemas.detections import (
    BoundingBox,
    CropCandidate,
    CropQuality,
    DecodeAttempt,
    DecodedSymbol,
    DetectionCandidate,
)


def test_preview_writer_includes_sampling_metadata(
    pipeline_context,
    in_memory_storage,
) -> None:
    pipeline_context.frames_processed = 3
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=0,
            timestamp_ms=0,
            original_width=1920,
            original_height=1080,
            processed_width=1080,
            processed_height=1920,
            orientation_applied="rotate_90_ccw",
            debug_frame_key="outputs/job-123/debug/frames/frame_000001.jpg",
        )
    ]
    pipeline_context.debug_frame_keys = [
        "outputs/job-123/debug/frames/frame_000001.jpg"
    ]
    pipeline_context.debug_overlay_keys = [
        "outputs/job-123/debug/overlays/frame_000001_detections.jpg"
    ]
    pipeline_context.debug_crop_keys = [
        "outputs/job-123/debug/crops/frame_000001_det_000001.jpg"
    ]
    pipeline_context.config["crop_extraction"] = {
        "top_crops_preview_limit": 20,
    }
    pipeline_context.config["barcode_qr_decode"] = {
        "max_payload_preview_length": 50,
    }
    pipeline_context.detections = [
        DetectionCandidate(
            detection_id="frame_000000_candidate_001_01",
            frame_index=0,
            timestamp_ms=0,
            label="price_tag_candidate",
            bbox=BoundingBox(x_min=10, y_min=20, x_max=120, y_max=80),
            confidence=0.88,
            source="heuristic_color_geometry_v1",
            attributes={"rank": 1},
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="frame_000000_crop_000001",
            detection_id="frame_000000_candidate_001_01",
            frame_index=0,
            timestamp_ms=0,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=120, y_max=80),
            padded_bbox=BoundingBox(x_min=8, y_min=18, x_max=122, y_max=82),
            crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
            width=114,
            height=64,
            quality=CropQuality(
                sharpness=220.0,
                brightness=190.0,
                contrast=48.0,
                glare_ratio=0.02,
                area_ratio=0.01,
                score=0.82,
            ),
            source="crop_extraction_v1",
            attributes={},
        )
    ]
    pipeline_context.decode_attempts = [
        DecodeAttempt(
            attempt_id="frame_000000_crop_000001__opencv_qr_detector__original",
            crop_id="frame_000000_crop_000001",
            decoder="opencv_qr_detector",
            variant="original",
            success=True,
            duration_ms=4,
        )
    ]
    pipeline_context.decoded_symbols = [
        DecodedSymbol(
            symbol_id="frame_000000_crop_000001__symbol_01",
            crop_id="frame_000000_crop_000001",
            detection_id="frame_000000_candidate_001_01",
            frame_index=0,
            timestamp_ms=0,
            symbol_type="qr",
            decoder="opencv_qr_detector",
            variant="resized_x2",
            payload="https://example.test/preview",
            confidence=0.77,
            bbox=BoundingBox(x_min=5, y_min=5, x_max=60, y_max=60),
        )
    ]

    outcome = PreviewWriterStage().run(pipeline_context)

    preview_key = outcome.output_summary["preview_key"]
    payload, _ = in_memory_storage.objects[preview_key]
    preview = json.loads(payload.decode("utf-8"))

    assert preview["frames_processed"] == 3
    assert preview["sampled_frames_count"] == 1
    assert preview["debug_frame_keys"] == [
        "outputs/job-123/debug/frames/frame_000001.jpg"
    ]
    assert preview["debug_overlay_keys"] == [
        "outputs/job-123/debug/overlays/frame_000001_detections.jpg"
    ]
    assert preview["debug_crop_keys"] == [
        "outputs/job-123/debug/crops/frame_000001_det_000001.jpg"
    ]
    assert preview["sampled_frames"][0]["orientation_applied"] == "rotate_90_ccw"
    assert preview["video_metadata"]["filename"] == "input.mp4"
    assert preview["rows_count"] == 0
    assert preview["detections_count"] == 1
    assert preview["detections_by_frame"] == {"0": 1}
    assert preview["sample_detections"][0]["label"] == "price_tag_candidate"
    assert preview["crops_count"] == 1
    assert preview["sample_crops"][0]["crop_id"] == "frame_000000_crop_000001"
    assert preview["sample_crops"][0]["quality"]["score"] == 0.82
    assert preview["decode_attempts_count"] == 1
    assert preview["decoded_symbols_count"] == 1
    assert preview["decoded_symbols_by_type"] == {"qr": 1, "barcode": 0, "unknown": 0}
    assert preview["sample_decoded_symbols"][0]["symbol_type"] == "qr"
    assert preview["sample_decoded_symbols"][0]["payload_preview"] == "https://example.test/preview"
