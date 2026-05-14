from __future__ import annotations

import json

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.preview_writer import PreviewWriterStage


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

    outcome = PreviewWriterStage().run(pipeline_context)

    preview_key = outcome.output_summary["preview_key"]
    payload, _ = in_memory_storage.objects[preview_key]
    preview = json.loads(payload.decode("utf-8"))

    assert preview["frames_processed"] == 3
    assert preview["sampled_frames_count"] == 1
    assert preview["debug_frame_keys"] == [
        "outputs/job-123/debug/frames/frame_000001.jpg"
    ]
    assert preview["sampled_frames"][0]["orientation_applied"] == "rotate_90_ccw"
    assert preview["video_metadata"]["filename"] == "input.mp4"
    assert preview["rows_count"] == 0
    assert preview["detections_count"] == 0
