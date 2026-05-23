from __future__ import annotations

from app.pipelines.price_tag_v2.stages.top_k_crop_selection import TopKCropSelectionStage
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality


def test_top_k_temporal_diversity_keeps_time_spread(pipeline_context) -> None:
    pipeline_context.config["top_k_crop_selection"] = {
        "enabled": True,
        "max_crops_per_track": 4,
        "max_total_crops": 10,
        "min_quality_score": 0.0,
        "temporal_diversity_enabled": True,
        "temporal_diversity_buckets": 4,
    }
    pipeline_context.crop_candidates = [
        _crop("early_0", frame_index=0, timestamp_ms=0, score=1.00),
        _crop("early_1", frame_index=1, timestamp_ms=10, score=0.99),
        _crop("early_2", frame_index=2, timestamp_ms=20, score=0.98),
        _crop("early_3", frame_index=3, timestamp_ms=30, score=0.97),
        _crop("late_100", frame_index=100, timestamp_ms=1000, score=0.60),
        _crop("late_200", frame_index=200, timestamp_ms=2000, score=0.59),
        _crop("late_300", frame_index=300, timestamp_ms=3000, score=0.58),
    ]

    outcome = TopKCropSelectionStage().run(pipeline_context)

    selected_ids = [crop.crop_id for crop in pipeline_context.crop_candidates]
    assert selected_ids == ["early_0", "late_100", "late_200", "late_300"]
    assert pipeline_context.artifacts["selected_crop_ids_by_track"]["track_a"] == [
        "early_0",
        "late_100",
        "late_200",
        "late_300",
    ]
    assert outcome.output_summary["temporal_diversity_enabled"] is True


def test_top_k_quality_order_is_default_without_temporal_diversity(
    pipeline_context,
) -> None:
    pipeline_context.config["top_k_crop_selection"] = {
        "enabled": True,
        "max_crops_per_track": 4,
        "max_total_crops": 10,
        "min_quality_score": 0.0,
        "temporal_diversity_enabled": False,
    }
    pipeline_context.crop_candidates = [
        _crop("early_0", frame_index=0, timestamp_ms=0, score=1.00),
        _crop("early_1", frame_index=1, timestamp_ms=10, score=0.99),
        _crop("early_2", frame_index=2, timestamp_ms=20, score=0.98),
        _crop("early_3", frame_index=3, timestamp_ms=30, score=0.97),
        _crop("late_100", frame_index=100, timestamp_ms=1000, score=0.60),
    ]

    TopKCropSelectionStage().run(pipeline_context)

    assert [crop.crop_id for crop in pipeline_context.crop_candidates] == [
        "early_0",
        "early_1",
        "early_2",
        "early_3",
    ]


def _crop(
    crop_id: str,
    *,
    frame_index: int,
    timestamp_ms: int,
    score: float,
) -> CropCandidate:
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    return CropCandidate(
        crop_id=crop_id,
        detection_id="det_track_a",
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        bbox=bbox,
        padded_bbox=bbox,
        crop_key="",
        width=10,
        height=10,
        quality=CropQuality(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            glare_ratio=0.0,
            area_ratio=0.0,
            score=score,
        ),
        source="test",
        attributes={"track_id": "track_a", "detection_confidence": score},
    )
