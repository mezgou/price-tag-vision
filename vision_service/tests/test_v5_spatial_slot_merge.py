from __future__ import annotations

from app.pipelines.price_tag_v5.stages.spatial_slot_merge import (
    SpatialSlotMergeConfig,
    V5SpatialSlotMergeStage,
    build_spatial_slot_assignments,
)
from app.schemas.detections import (
    BoundingBox,
    CropCandidate,
    CropQuality,
    DetectionCandidate,
)


def test_spatial_slot_assignment_merges_temporally_adjacent_tracklets() -> None:
    config = SpatialSlotMergeConfig.from_context(_ctx_config())
    detections = [
        _det("track_a", "a1", 0, 0, (100, 100, 220, 180)),
        _det("track_a", "a2", 4, 200, (102, 101, 222, 181)),
        _det("track_b", "b1", 12, 700, (106, 102, 226, 182)),
        _det("track_b", "b2", 16, 900, (108, 103, 228, 183)),
    ]

    track_to_slot, slots, edges = build_spatial_slot_assignments(
        detections,
        config=config,
        camera=None,
    )

    assert track_to_slot["track_a"] == track_to_slot["track_b"]
    assert len(slots) == 1
    assert edges


def test_spatial_slot_assignment_keeps_overlapping_tracks_separate() -> None:
    config = SpatialSlotMergeConfig.from_context(_ctx_config())
    detections = [
        _det("track_a", "a1", 0, 0, (100, 100, 220, 180)),
        _det("track_a", "a2", 4, 1000, (102, 101, 222, 181)),
        _det("track_b", "b1", 2, 500, (106, 102, 226, 182)),
        _det("track_b", "b2", 6, 1200, (108, 103, 228, 183)),
    ]

    track_to_slot, slots, edges = build_spatial_slot_assignments(
        detections,
        config=config,
        camera=None,
    )

    assert track_to_slot["track_a"] != track_to_slot["track_b"]
    assert len(slots) == 2
    assert edges == []


def test_spatial_slot_assignment_absorbs_overlapping_untracked_singleton() -> None:
    config = SpatialSlotMergeConfig.from_context(_ctx_config())
    detections = [
        _det("track_a", "a1", 0, 0, (100, 100, 220, 180)),
        _det("track_a", "a2", 4, 1000, (102, 101, 222, 181)),
        _det("untracked_000001", "u1", 2, 500, (104, 102, 224, 182)),
    ]

    track_to_slot, slots, edges = build_spatial_slot_assignments(
        detections,
        config=config,
        camera=None,
    )

    assert track_to_slot["track_a"] == track_to_slot["untracked_000001"]
    assert len(slots) == 1
    assert edges[0].reason == "overlap_untracked_absorb"


def test_spatial_slot_stage_rewrites_tracks_and_preserves_sources(
    pipeline_context,
) -> None:
    pipeline_context.config.update(_ctx_config().config)
    pipeline_context.detections = [
        _det("track_a", "a1", 0, 0, (100, 100, 220, 180)),
        _det("track_b", "b1", 12, 700, (106, 102, 226, 182)),
    ]
    pipeline_context.crop_candidates = [
        _crop("crop_a", "a1", "track_a", 0, (100, 100, 220, 180)),
        _crop("crop_b", "b1", "track_b", 700, (106, 102, 226, 182)),
    ]

    outcome = V5SpatialSlotMergeStage().run(pipeline_context)

    slot_ids = {d.attributes["track_id"] for d in pipeline_context.detections}
    assert len(slot_ids) == 1
    assert {d.attributes["source_track_id"] for d in pipeline_context.detections} == {
        "track_a",
        "track_b",
    }
    assert {c.attributes["track_id"] for c in pipeline_context.crop_candidates} == slot_ids
    assert outcome.output_summary["input_tracks"] == 2
    assert outcome.output_summary["slots"] == 1
    assert outcome.output_summary["merged_components"] == 1


def _ctx_config():
    class _Context:
        config = {
            "v5_spatial_slot_merge": {
                "enabled": True,
                "max_time_gap_ms": 1600,
                "min_edge_score": 0.72,
                "min_iou": 0.10,
                "max_center_distance_ratio": 0.55,
                "max_y_center_delta_ratio": 0.35,
                "min_size_ratio": 0.45,
                "allow_time_overlap": False,
                "absorb_untracked_overlaps": True,
                "overlap_absorb_min_iou": 0.55,
                "overlap_absorb_max_detections": 2,
            }
        }

    return _Context()


def _det(
    track_id: str,
    detection_id: str,
    frame_index: int,
    timestamp_ms: int,
    bbox: tuple[int, int, int, int],
) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        label="price_tag",
        bbox=BoundingBox(
            x_min=bbox[0],
            y_min=bbox[1],
            x_max=bbox[2],
            y_max=bbox[3],
        ),
        confidence=0.9,
        source="unit-test",
        attributes={"track_id": track_id},
    )


def _crop(
    crop_id: str,
    detection_id: str,
    track_id: str,
    timestamp_ms: int,
    bbox: tuple[int, int, int, int],
) -> CropCandidate:
    box = BoundingBox(
        x_min=bbox[0],
        y_min=bbox[1],
        x_max=bbox[2],
        y_max=bbox[3],
    )
    return CropCandidate(
        crop_id=crop_id,
        detection_id=detection_id,
        frame_index=0,
        timestamp_ms=timestamp_ms,
        bbox=box,
        padded_bbox=box,
        crop_key="",
        width=box.width,
        height=box.height,
        quality=CropQuality(
            sharpness=1.0,
            brightness=1.0,
            contrast=1.0,
            glare_ratio=0.0,
            area_ratio=0.01,
            score=0.9,
        ),
        source="unit-test",
        attributes={"track_id": track_id},
    )
