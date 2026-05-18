from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.datasets.yolo_price_tag_dataset import (
    bbox_to_yolo,
    build_yolo_price_tag_dataset,
    clip_bbox_to_frame,
    extract_frame_at_timestamp,
    group_rows_by_timestamp,
    parse_ground_truth_row,
)
from app.datasets.yolo_price_tag_dataset import FloatBoundingBox
from tests.video_factory import write_synthetic_video


def test_parse_ground_truth_row_normalizes_timestamp_and_bbox() -> None:
    parsed = parse_ground_truth_row(
        {
            "frame_timestamp": "1234,6",
            "x_min": "10,25",
            "y_min": "20",
            "x_max": "110,75",
            "y_max": "70,5",
            "id_sku": "sku-42",
            "barcode": "нет",
            "product_name": "Price tag",
        },
        row_index=3,
        video_stem="sample_video",
    )

    assert parsed.timestamp_ms == 1235
    assert parsed.bbox == FloatBoundingBox(10.25, 20.0, 110.75, 70.5)
    assert parsed.id_sku == "sku-42"
    assert parsed.barcode is None


def test_clip_bbox_to_frame_clips_partial_and_rejects_degenerate() -> None:
    partial = clip_bbox_to_frame(
        FloatBoundingBox(-5.0, 10.0, 35.0, 40.0),
        frame_width=30,
        frame_height=50,
    )
    outside = clip_bbox_to_frame(
        FloatBoundingBox(35.0, 10.0, 60.0, 40.0),
        frame_width=30,
        frame_height=50,
    )

    assert partial == FloatBoundingBox(0.0, 10.0, 30.0, 40.0)
    assert outside is None


def test_bbox_to_yolo_returns_expected_normalized_values() -> None:
    values = bbox_to_yolo(
        FloatBoundingBox(10.0, 20.0, 30.0, 60.0),
        image_width=100,
        image_height=200,
    )

    assert values == (0.2, 0.2, 0.2, 0.2)


def test_group_rows_by_timestamp_collapses_multiple_boxes() -> None:
    rows = [
        parse_ground_truth_row(
            {
                "frame_timestamp": "0",
                "x_min": "0",
                "y_min": "0",
                "x_max": "10",
                "y_max": "10",
                "id_sku": "1",
                "barcode": "",
                "product_name": "a",
            },
            row_index=1,
            video_stem="sample_video",
        ),
        parse_ground_truth_row(
            {
                "frame_timestamp": "0",
                "x_min": "10",
                "y_min": "10",
                "x_max": "20",
                "y_max": "20",
                "id_sku": "2",
                "barcode": "",
                "product_name": "b",
            },
            row_index=2,
            video_stem="sample_video",
        ),
        parse_ground_truth_row(
            {
                "frame_timestamp": "250",
                "x_min": "20",
                "y_min": "20",
                "x_max": "30",
                "y_max": "30",
                "id_sku": "3",
                "barcode": "",
                "product_name": "c",
            },
            row_index=3,
            video_stem="sample_video",
        ),
    ]

    grouped = group_rows_by_timestamp(rows)

    assert [group.timestamp_ms for group in grouped] == [0, 250]
    assert len(grouped[0].rows) == 2
    assert len(grouped[1].rows) == 1


def test_extract_frame_at_timestamp_reads_expected_synthetic_frame(tmp_path: Path) -> None:
    video_path = write_synthetic_video(
        tmp_path / "frames.mp4",
        width=64,
        height=48,
        fps=5.0,
        frame_count=5,
    )

    extracted = extract_frame_at_timestamp(video_path, 400)
    background_pixel = extracted.frame[0, 0].astype(float)

    assert extracted.frame_index == 2
    assert np.allclose(background_pixel, np.array([30.0, 60.0, 90.0]), atol=15.0)


def test_build_dataset_writes_expected_structure_and_report(tmp_path: Path) -> None:
    input_root = tmp_path / "videos"
    _create_labeled_video(
        input_root / "video_a",
        timestamps_to_boxes={
            0: [(20.0, 20.0, 120.0, 60.0), (140.0, 25.0, 220.0, 70.0)],
            200: [(40.0, 80.0, 150.0, 130.0)],
        },
    )
    _create_labeled_video(
        input_root / "video_b",
        timestamps_to_boxes={
            0: [(30.0, 30.0, 130.0, 90.0)],
        },
    )

    output_dir = tmp_path / "dataset_out"
    report = build_yolo_price_tag_dataset(
        input_root=input_root,
        output_dir=output_dir,
        orientation_mode="none",
        train_videos=["video_a"],
        val_videos=["video_b"],
        make_orientation_qa=True,
        orientation_review_note="none looks correct for the synthetic upright frames.",
        dataset_path_value="artifacts/datasets/test_price_tag_yolo",
    )

    dataset_yaml = yaml.safe_load((output_dir / "dataset.yaml").read_text(encoding="utf-8"))

    assert dataset_yaml == {
        "path": "artifacts/datasets/test_price_tag_yolo",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "price_tag"},
    }
    assert (output_dir / "images" / "train" / "video_a_ts_0.jpg").exists()
    assert (output_dir / "images" / "train" / "video_a_ts_200.jpg").exists()
    assert (output_dir / "images" / "val" / "video_b_ts_0.jpg").exists()
    assert (output_dir / "labels" / "train" / "video_a_ts_0.txt").exists()
    assert (output_dir / "labels" / "val" / "video_b_ts_0.txt").exists()
    assert (output_dir / "qa" / "overlays" / "video_a_ts_0.jpg").exists()
    assert (output_dir / "qa" / "contact_sheets" / "train_gt_overlays.jpg").exists()
    assert (output_dir / "qa" / "contact_sheets" / "val_gt_overlays.jpg").exists()
    assert (output_dir / "qa" / "contact_sheets" / "crops_gt_examples.jpg").exists()
    assert (output_dir / "qa" / "dataset_report.json").exists()
    for orientation_mode in ("none", "rotate_90_cw", "rotate_90_ccw", "rotate_180"):
        assert any((output_dir / "qa" / "orientation_check" / orientation_mode).glob("*.jpg"))

    assert report["total_rows"] == 4
    assert report["valid_boxes"] == 4
    assert report["skipped_boxes"] == 0
    assert report["images_count"] == 3
    assert report["train_images"] == 2
    assert report["val_images"] == 1
    assert report["boxes_per_video"] == {"video_a": 3, "video_b": 1}
    assert report["chosen_orientation_mode"] == "none"


def test_build_dataset_dry_run_reports_expected_counts_without_writing_files(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "videos"
    _create_labeled_video(
        input_root / "video_a",
        timestamps_to_boxes={
            0: [(20.0, 20.0, 120.0, 60.0)],
        },
    )
    _create_labeled_video(
        input_root / "video_b",
        timestamps_to_boxes={
            0: [(30.0, 30.0, 130.0, 90.0)],
            200: [(50.0, 100.0, 180.0, 150.0)],
        },
    )

    output_dir = tmp_path / "dataset_out"
    report = build_yolo_price_tag_dataset(
        input_root=input_root,
        output_dir=output_dir,
        orientation_mode="none",
        train_videos=["video_a"],
        val_videos=["video_b"],
        dry_run=True,
    )

    assert report["total_rows"] == 3
    assert report["valid_boxes"] == 3
    assert report["skipped_boxes"] == 0
    assert report["images_count"] == 3
    assert not output_dir.exists()


def test_build_dataset_propagates_labels_to_neighbor_frames(tmp_path: Path) -> None:
    input_root = tmp_path / "videos"
    video_dir = input_root / "video_a"
    video_dir.mkdir(parents=True)
    _write_template_video(video_dir / "video_a.mp4")
    _write_csv(
        video_dir / "video_a.csv",
        [
            {
                "frame_timestamp": "400",
                "x_min": "20",
                "y_min": "20",
                "x_max": "70",
                "y_max": "55",
                "id_sku": "sku-1",
                "barcode": "",
                "product_name": "item",
            }
        ],
    )
    _create_labeled_video(
        input_root / "video_b",
        timestamps_to_boxes={0: [(10.0, 10.0, 40.0, 40.0)]},
    )

    output_dir = tmp_path / "dataset_out"
    report = build_yolo_price_tag_dataset(
        input_root=input_root,
        output_dir=output_dir,
        orientation_mode="none",
        train_videos=["video_a"],
        val_videos=["video_b"],
        propagate_frames=1,
        propagation_ncc_threshold=0.8,
    )

    assert report["keyframe_images"] == 2
    assert report["propagated_images"] == 2
    assert report["propagated_boxes"] == 2
    assert report["images_count"] == 4
    assert len(list((output_dir / "labels" / "train").glob("*.txt"))) == 3
    assert len(list((output_dir / "labels" / "val").glob("*.txt"))) == 1


def test_build_dataset_can_hold_out_keyframes_and_stride_propagation(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "videos"
    video_dir = input_root / "video_a"
    video_dir.mkdir(parents=True)
    _write_template_video(video_dir / "video_a.mp4")
    _write_csv(
        video_dir / "video_a.csv",
        [
            {
                "frame_timestamp": "0",
                "x_min": "20",
                "y_min": "20",
                "x_max": "70",
                "y_max": "55",
                "id_sku": "sku-1",
                "barcode": "",
                "product_name": "item-a",
            },
            {
                "frame_timestamp": "400",
                "x_min": "20",
                "y_min": "20",
                "x_max": "70",
                "y_max": "55",
                "id_sku": "sku-2",
                "barcode": "",
                "product_name": "item-b",
            },
            {
                "frame_timestamp": "800",
                "x_min": "20",
                "y_min": "20",
                "x_max": "70",
                "y_max": "55",
                "id_sku": "sku-3",
                "barcode": "",
                "product_name": "item-c",
            },
        ],
    )
    _create_labeled_video(
        input_root / "video_b",
        timestamps_to_boxes={0: [(10.0, 10.0, 40.0, 40.0)]},
    )

    output_dir = tmp_path / "dataset_out"
    report = build_yolo_price_tag_dataset(
        input_root=input_root,
        output_dir=output_dir,
        orientation_mode="none",
        train_videos=["video_a"],
        val_videos=["video_b"],
        propagate_frames=2,
        propagation_frame_stride=2,
        keyframe_holdout_stride=2,
        propagation_ncc_threshold=0.8,
    )

    assert report["train_images"] == 2
    assert report["val_images"] == 2
    assert report["propagated_images"] == 0
    assert report["propagation_frame_stride"] == 2
    assert report["keyframe_holdout_stride"] == 2
    assert report["keyframe_holdout_by_video"] == {"video_a": [400]}


def _create_labeled_video(
    directory: Path,
    *,
    timestamps_to_boxes: dict[int, list[tuple[float, float, float, float]]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    video_path = write_synthetic_video(
        directory / f"{directory.name}.mp4",
        width=320,
        height=180,
        fps=5.0,
        frame_count=8,
    )
    csv_path = directory / f"{directory.name}.csv"
    rows: list[dict[str, str]] = []
    row_index = 1
    for timestamp_ms, boxes in timestamps_to_boxes.items():
        for box in boxes:
            rows.append(
                {
                    "frame_timestamp": str(timestamp_ms),
                    "x_min": f"{box[0]}",
                    "y_min": f"{box[1]}",
                    "x_max": f"{box[2]}",
                    "y_max": f"{box[3]}",
                    "id_sku": f"sku-{directory.name}-{row_index}",
                    "barcode": "",
                    "product_name": f"item-{row_index}",
                }
            )
            row_index += 1
    _write_csv(csv_path, rows)
    assert video_path.exists()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "frame_timestamp",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "id_sku",
        "barcode",
        "product_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_template_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (120, 80))
    assert writer.isOpened()
    try:
        for _index in range(5):
            frame = np.full((80, 120, 3), 240, dtype=np.uint8)
            cv2.rectangle(frame, (20, 20), (70, 55), (20, 20, 20), thickness=-1)
            cv2.putText(
                frame,
                "A",
                (32, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
