from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipelines.price_tag_cpu_v1.orientation import OrientationMode, apply_orientation

FrameArray = NDArray[np.uint8]
CONTACT_SHEET_BACKGROUND = (245, 245, 245)
CONTACT_SHEET_TEXT = (24, 24, 24)
CONTACT_SHEET_ACCENT = (0, 160, 220)
OVERLAY_TEXT_COLOR = (20, 20, 20)
OVERLAY_TEXT_BG = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class YoloRunPaths:
    project_dir: Path
    run_dir: Path
    weights_dir: Path
    qa_dir: Path
    val_prediction_overlays_dir: Path
    real_video_overlays_dir: Path
    unlabeled_overlays_dir: Path
    export_dir: Path
    evaluation_report_path: Path
    val_eval_dir: Path


@dataclass(frozen=True, slots=True)
class ContactSheetItem:
    title: str
    subtitle: str
    image: FrameArray


@dataclass(frozen=True, slots=True)
class SampledVideoFrame:
    frame: FrameArray
    source_path: Path
    frame_index: int
    timestamp_ms: int
    display_name: str


def resolve_yolo_run_paths(project_dir: Path, run_name: str) -> YoloRunPaths:
    run_dir = project_dir / run_name
    qa_dir = run_dir / "qa"
    return YoloRunPaths(
        project_dir=project_dir,
        run_dir=run_dir,
        weights_dir=run_dir / "weights",
        qa_dir=qa_dir,
        val_prediction_overlays_dir=qa_dir / "val_prediction_overlays",
        real_video_overlays_dir=qa_dir / "real_video_prediction_overlays",
        unlabeled_overlays_dir=qa_dir / "unlabeled_prediction_overlays",
        export_dir=run_dir / "export",
        evaluation_report_path=qa_dir / "evaluation_report.json",
        val_eval_dir=run_dir / "val_eval",
    )


def infer_run_dir_from_model_path(model_path: Path) -> Path:
    if model_path.parent.name != "weights":
        raise ValueError(
            f"Expected model path under a weights directory, got: {model_path}"
        )
    return model_path.parent.parent


def build_train_overrides(
    *,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    patience: int,
    seed: int,
    cache: bool,
    plots: bool,
    workers: int,
) -> dict[str, Any]:
    return {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "patience": patience,
        "seed": seed,
        "cache": cache,
        "plots": plots,
        "workers": workers,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        # Optimiser / schedule tuned for the propagated price-tag dataset.
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
        "close_mosaic": 10,
        # Augmentation profile (doc 5.3): fight glare, slight camera tilt,
        # and simulate an adjacent price tag.
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


def build_val_overrides(
    *,
    data: str,
    imgsz: int,
    conf: float,
    iou: float,
    project: str,
    name: str,
    device: str,
    plots: bool,
    workers: int,
) -> dict[str, Any]:
    return {
        "data": data,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "project": project,
        "name": name,
        "device": device,
        "plots": plots,
        "workers": workers,
        "exist_ok": True,
        "save_json": False,
        "verbose": True,
    }


def count_label_boxes(labels_dir: Path) -> int:
    total = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        total += len(lines)
    return total


def count_split_images(images_dir: Path) -> int:
    return len(sorted(images_dir.glob("*.jpg")))


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_evaluation_report(
    *,
    model_path: Path,
    dataset_path: Path,
    run_dir: Path,
    train_images: int,
    val_images: int,
    train_boxes: int,
    val_boxes: int,
    precision: float | None,
    recall: float | None,
    map50: float | None,
    map50_95: float | None,
    best_conf_suggestion: float,
    notes: list[str],
    device_requested: str,
    device_used: str,
    training_duration_sec: float | None,
    val_output_dir: Path,
    results_csv_path: Path | None,
    results_png_path: Path | None,
    export_status: dict[str, Any] | None = None,
    qa_artifacts: dict[str, Any] | None = None,
    dataset_issues: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "run_dir": str(run_dir),
        "train_images": train_images,
        "val_images": val_images,
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "precision": _rounded_metric(precision),
        "recall": _rounded_metric(recall),
        "mAP50": _rounded_metric(map50),
        "mAP50_95": _rounded_metric(map50_95),
        "best_conf_suggestion": _rounded_metric(best_conf_suggestion),
        "device_requested": device_requested,
        "device_used": device_used,
        "training_duration_sec": _rounded_metric(training_duration_sec),
        "val_output_dir": str(val_output_dir),
        "results_csv_path": None if results_csv_path is None else str(results_csv_path),
        "results_png_path": None if results_png_path is None else str(results_png_path),
        "notes": notes,
        "export_status": export_status or {},
        "qa_artifacts": qa_artifacts or {},
        "dataset_issues": dataset_issues or {},
    }


def sample_video_frames(
    *,
    video_path: Path,
    sample_fps: float,
    max_frames: int,
    orientation_mode: OrientationMode,
) -> list[SampledVideoFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}")

    sampled_frames: list[SampledVideoFrame] = []
    try:
        source_fps = _positive_capture_float(capture.get(cv2.CAP_PROP_FPS))
        sample_interval = _resolve_sample_interval(source_fps=source_fps, sample_fps=sample_fps)

        frame_index = 0
        while len(sampled_frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None:
                frame_index += 1
                continue
            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            oriented_frame = apply_orientation(frame, orientation_mode)
            timestamp_ms = _frame_timestamp_ms(frame_index=frame_index, source_fps=source_fps)
            sampled_frames.append(
                SampledVideoFrame(
                    frame=oriented_frame,
                    source_path=video_path,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms if timestamp_ms is not None else 0,
                    display_name=f"{video_path.stem}_frame_{frame_index:06d}_ts_{timestamp_ms or 0}",
                )
            )
            frame_index += 1
    finally:
        capture.release()

    return sampled_frames


def discover_unlabeled_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    preferred_order = ["25_12-20.mp4", "26_12-20.mp4", "26_2-10.mp4"]
    by_name = {path.name: path for path in root.glob("*.mp4")}
    ordered = [by_name[name] for name in preferred_order if name in by_name]
    remaining = sorted(path for path in root.glob("*.mp4") if path.name not in by_name or path not in ordered)
    return ordered + remaining


def build_contact_sheet_image(
    *,
    items: list[ContactSheetItem],
    thumb_width: int,
    thumb_height: int,
) -> FrameArray:
    if not items:
        return np.full((96, 96, 3), CONTACT_SHEET_BACKGROUND, dtype=np.uint8)

    padding = 12
    text_block_height = 42
    columns = min(4, max(1, int(math.ceil(math.sqrt(len(items))))))
    rows = int(math.ceil(len(items) / columns))
    cell_width = thumb_width + (padding * 2)
    cell_height = thumb_height + text_block_height + (padding * 2)
    canvas = np.full(
        (rows * cell_height, columns * cell_width, 3),
        CONTACT_SHEET_BACKGROUND,
        dtype=np.uint8,
    )

    for item_index, item in enumerate(items):
        row_index = item_index // columns
        column_index = item_index % columns
        origin_x = column_index * cell_width
        origin_y = row_index * cell_height
        text_origin_y = origin_y + padding + 14

        cv2.putText(
            canvas,
            item.title,
            (origin_x + padding, text_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            CONTACT_SHEET_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            item.subtitle,
            (origin_x + padding, text_origin_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            CONTACT_SHEET_TEXT,
            1,
            cv2.LINE_AA,
        )
        thumbnail = _fit_image_to_box(
            image=item.image,
            target_width=thumb_width,
            target_height=thumb_height,
        )
        thumb_origin_y = origin_y + padding + text_block_height
        thumb_origin_x = origin_x + padding
        canvas[
            thumb_origin_y : thumb_origin_y + thumb_height,
            thumb_origin_x : thumb_origin_x + thumb_width,
        ] = thumbnail
        cv2.rectangle(
            canvas,
            (thumb_origin_x, thumb_origin_y),
            (thumb_origin_x + thumb_width - 1, thumb_origin_y + thumb_height - 1),
            CONTACT_SHEET_ACCENT,
            1,
        )

    return canvas


def save_contact_sheet(
    *,
    path: Path,
    items: list[ContactSheetItem],
    thumb_width: int,
    thumb_height: int,
    quality: int = 92,
) -> None:
    image = build_contact_sheet_image(
        items=items,
        thumb_width=thumb_width,
        thumb_height=thumb_height,
    )
    save_jpeg(path=path, image=image, quality=quality)


def save_jpeg(path: Path, image: FrameArray, *, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        raise RuntimeError(f"Failed to encode JPEG: {path}")
    path.write_bytes(encoded.tobytes())


def add_overlay_text(
    image: FrameArray,
    *,
    title: str,
    subtitle: str | None = None,
) -> FrameArray:
    annotated = image.copy()
    font_scale = max(min(annotated.shape[0], annotated.shape[1]) / 2400.0, 0.5)
    thickness = max(int(round(font_scale * 2)), 1)
    _draw_text_box(
        annotated,
        title,
        origin=(18, 26),
        font_scale=font_scale,
        thickness=thickness,
    )
    if subtitle:
        _draw_text_box(
            annotated,
            subtitle,
            origin=(18, 52),
            font_scale=font_scale,
            thickness=thickness,
        )
    return annotated


def format_conf_tag(confidence: float) -> str:
    return f"{int(round(confidence * 100)):03d}"


def summarize_detection_counts(counts: list[int]) -> dict[str, float | int]:
    if not counts:
        return {"frames": 0, "detections_total": 0, "detections_mean": 0.0}
    return {
        "frames": len(counts),
        "detections_total": int(sum(counts)),
        "detections_mean": round(statistics.fmean(counts), 6),
        "detections_max": int(max(counts)),
    }


def _fit_image_to_box(
    *,
    image: FrameArray,
    target_width: int,
    target_height: int,
) -> FrameArray:
    canvas = np.full(
        (target_height, target_width, 3),
        CONTACT_SHEET_BACKGROUND,
        dtype=np.uint8,
    )
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        return canvas

    scale = min(target_width / float(source_width), target_height / float(source_height))
    resized_width = max(int(round(source_width * scale)), 1)
    resized_height = max(int(round(source_height * scale)), 1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _draw_text_box(
    image: FrameArray,
    text: str,
    *,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    origin_x, origin_y = origin
    top_left = (max(origin_x - 4, 0), max(origin_y - text_height - baseline - 4, 0))
    bottom_right = (
        min(origin_x + text_width + 4, image.shape[1] - 1),
        min(origin_y + baseline + 4, image.shape[0] - 1),
    )
    cv2.rectangle(image, top_left, bottom_right, OVERLAY_TEXT_BG, thickness=-1)
    cv2.putText(
        image,
        text,
        (origin_x, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        OVERLAY_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def _positive_capture_float(value: float) -> float | None:
    if value <= 0:
        return None
    return float(value)


def _resolve_sample_interval(*, source_fps: float | None, sample_fps: float) -> int:
    if source_fps is None or source_fps <= 0:
        return 1
    return max(int(round(source_fps / sample_fps)), 1)


def _frame_timestamp_ms(*, frame_index: int, source_fps: float | None) -> int | None:
    if source_fps is None or source_fps <= 0:
        return None
    return int(round((frame_index / source_fps) * 1000))


def _rounded_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
