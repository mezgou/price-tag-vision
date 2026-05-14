from __future__ import annotations

from app.utils.video_metadata import read_video_metadata


def test_read_video_metadata_returns_graceful_fallback_for_missing_file(
    tmp_path,
) -> None:
    missing_video_path = tmp_path / "missing.mp4"

    metadata, warnings = read_video_metadata(
        missing_video_path,
        input_video_key="inputs/job-123/missing.mp4",
    )

    assert metadata.filename == "missing.mp4"
    assert metadata.input_video_key == "inputs/job-123/missing.mp4"
    assert metadata.duration_ms is None
    assert metadata.fps is None
    assert metadata.frame_count is None
    assert metadata.width is None
    assert metadata.height is None
    assert warnings
