from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_cpu_v1.stages.frame_metadata import FrameMetadataStage
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import FrameSamplingStage
from tests.video_factory import write_synthetic_video


def test_frame_sampling_stage_returns_sampled_frame_metadata(
    pipeline_context,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    video_path = write_synthetic_video(tmp_path / "inputs" / "sample.mp4")
    pipeline_context.local_video_path = video_path
    pipeline_context.input_video_key = "inputs/job-123/sample.mp4"
    pipeline_context.config["frame_sampling"] = {
        "sample_fps": 1.0,
        "max_frames": 20,
        "orientation_mode": "rotate_90_ccw",
        "debug_save_frames": True,
        "debug_jpeg_quality": 85,
    }

    FrameMetadataStage().run(pipeline_context)
    outcome = FrameSamplingStage().run(pipeline_context)

    assert outcome.output_summary["frames_processed"] == 2
    assert outcome.output_summary["sampled_frames_count"] == 2
    assert outcome.output_summary["debug_frames_saved"] == 2
    assert pipeline_context.frames_processed == 2
    assert len(pipeline_context.sampled_frames) == 2
    assert len(pipeline_context.debug_frame_keys) == 2
    assert pipeline_context.sampled_frames[0].to_dict() == {
        "frame_index": 0,
        "timestamp_ms": 0,
        "original_width": 320,
        "original_height": 180,
        "processed_width": 180,
        "processed_height": 320,
        "orientation_applied": "rotate_90_ccw",
        "debug_frame_key": "outputs/job-123/debug/frames/frame_000001.jpg",
    }
    assert pipeline_context.sampled_frames[1].frame_index == 5
    assert pipeline_context.sampled_frames[1].timestamp_ms == 1000
    assert pipeline_context.artifacts["debug_frame_keys"] == [
        "outputs/job-123/debug/frames/frame_000001.jpg",
        "outputs/job-123/debug/frames/frame_000002.jpg",
    ]
    assert sorted(in_memory_storage.objects) == [
        "outputs/job-123/debug/frames/frame_000001.jpg",
        "outputs/job-123/debug/frames/frame_000002.jpg",
    ]
