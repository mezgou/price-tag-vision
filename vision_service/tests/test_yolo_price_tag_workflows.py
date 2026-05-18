from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.yolo.price_tag_workflows import (
    ContactSheetItem,
    build_contact_sheet_image,
    build_evaluation_report,
    build_train_overrides,
    infer_run_dir_from_model_path,
    resolve_yolo_run_paths,
    sample_video_frames,
    write_json_report,
)
from tests.video_factory import write_synthetic_video


def test_build_train_overrides_uses_expected_defaults() -> None:
    overrides = build_train_overrides(
        data="artifacts/datasets/price_tag_yolo_v1/dataset.yaml",
        epochs=120,
        imgsz=1280,
        batch=2,
        device="0",
        project="artifacts/models",
        name="price_tag_yolo11n_v1",
        patience=25,
        seed=42,
        cache=False,
        plots=True,
        workers=0,
    )

    assert overrides == {
        "data": "artifacts/datasets/price_tag_yolo_v1/dataset.yaml",
        "epochs": 120,
        "imgsz": 1280,
        "batch": 2,
        "device": "0",
        "project": "artifacts/models",
        "name": "price_tag_yolo11n_v1",
        "patience": 25,
        "seed": 42,
        "cache": False,
        "plots": True,
        "workers": 0,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 8.0,
        "translate": 0.10,
        "scale": 0.5,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.8,
        "mixup": 0.1,
        "copy_paste": 0.2,
    }


def test_resolve_yolo_run_paths_and_infer_run_dir_are_consistent(tmp_path: Path) -> None:
    paths = resolve_yolo_run_paths(tmp_path / "models", "price_tag_yolo11n_v1")
    inferred = infer_run_dir_from_model_path(paths.weights_dir / "best.pt")

    assert paths.run_dir == tmp_path / "models" / "price_tag_yolo11n_v1"
    assert paths.export_dir == paths.run_dir / "export"
    assert paths.evaluation_report_path == paths.run_dir / "qa" / "evaluation_report.json"
    assert inferred == paths.run_dir


def test_write_json_report_persists_evaluation_payload(tmp_path: Path) -> None:
    report = build_evaluation_report(
        model_path=tmp_path / "weights" / "best.pt",
        dataset_path=tmp_path / "dataset.yaml",
        run_dir=tmp_path,
        train_images=25,
        val_images=2,
        train_boxes=128,
        val_boxes=29,
        precision=0.81,
        recall=0.73,
        map50=0.79,
        map50_95=0.51,
        best_conf_suggestion=0.1,
        notes=["qa required"],
        device_requested="0",
        device_used="cpu",
        training_duration_sec=123.456,
        val_output_dir=tmp_path / "val_eval",
        results_csv_path=tmp_path / "results.csv",
        results_png_path=tmp_path / "results.png",
    )
    path = tmp_path / "qa" / "evaluation_report.json"
    write_json_report(path, report)

    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["precision"] == 0.81
    assert saved["device_used"] == "cpu"
    assert saved["training_duration_sec"] == 123.456
    assert saved["notes"] == ["qa required"]


def test_build_contact_sheet_image_handles_empty_items() -> None:
    sheet = build_contact_sheet_image(items=[], thumb_width=200, thumb_height=120)

    assert sheet.shape == (96, 96, 3)
    assert np.all(sheet == 245)


def test_sample_video_frames_applies_orientation(tmp_path: Path) -> None:
    video_path = write_synthetic_video(
        tmp_path / "oriented.mp4",
        width=80,
        height=40,
        fps=5.0,
        frame_count=4,
    )

    frames = sample_video_frames(
        video_path=video_path,
        sample_fps=1.0,
        max_frames=2,
        orientation_mode="rotate_90_ccw",
    )

    assert len(frames) == 1
    assert frames[0].frame.shape[:2] == (80, 40)
    assert frames[0].frame_index == 0
    assert frames[0].timestamp_ms == 0


def test_build_contact_sheet_image_renders_items() -> None:
    image = np.full((40, 80, 3), 127, dtype=np.uint8)
    sheet = build_contact_sheet_image(
        items=[ContactSheetItem(title="frame", subtitle="det=1", image=image)],
        thumb_width=100,
        thumb_height=80,
    )

    assert sheet.shape[0] > 80
    assert sheet.shape[1] > 100
