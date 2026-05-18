from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from app.pipelines.price_tag_cpu_v1.orientation import (
    SUPPORTED_ORIENTATION_MODES,
    OrientationMode,
    apply_orientation,
)
from app.utils.camera import CameraModel


def _transform_frame(
    frame: FrameArray,
    *,
    orientation_mode: OrientationMode,
    camera: CameraModel | None,
) -> FrameArray:
    if camera is not None:
        return camera.raw_frame_to_processed(frame)
    return apply_orientation(frame, orientation_mode)


def _transform_bbox(
    bbox: "FloatBoundingBox",
    *,
    orientation_mode: OrientationMode,
    source_width: int,
    source_height: int,
    camera: CameraModel | None,
) -> "FloatBoundingBox":
    if camera is not None:
        x1, y1, x2, y2 = camera.raw_box_to_processed(
            (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
        )
        return FloatBoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
    return rotate_bbox(
        bbox,
        mode=orientation_mode,
        source_width=source_width,
        source_height=source_height,
    )

FrameArray = NDArray[np.uint8]
CLASS_ID = 0
CLASS_NAME = "price_tag"
OVERLAY_BOX_COLOR = (0, 210, 255)
OVERLAY_TEXT_COLOR = (20, 20, 20)
OVERLAY_TEXT_BG = (255, 255, 255)
CONTACT_SHEET_BACKGROUND = (245, 245, 245)
CONTACT_SHEET_TEXT = (24, 24, 24)
CONTACT_SHEET_ACCENT = (0, 160, 220)
DEFAULT_JPEG_QUALITY = 95
PLACEHOLDER_VALUES = {"", "-", "нет", "none", "n/a"}


@dataclass(frozen=True, slots=True)
class FloatBoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            raise ValueError("Bounding box height must be positive.")
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class GroundTruthRow:
    row_index: int
    video_stem: str
    timestamp_ms: int
    bbox: FloatBoundingBox
    id_sku: str | None
    barcode: str | None
    product_name: str | None


@dataclass(frozen=True, slots=True)
class TimestampAnnotationGroup:
    video_stem: str
    timestamp_ms: int
    rows: tuple[GroundTruthRow, ...]


@dataclass(frozen=True, slots=True)
class LabeledVideoSource:
    video_stem: str
    video_path: Path
    csv_path: Path


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    frame: FrameArray
    frame_index: int | None
    source_fps: float | None


@dataclass(frozen=True, slots=True)
class ValidAnnotation:
    row: GroundTruthRow
    bbox: FloatBoundingBox
    yolo: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ContactSheetItem:
    title: str
    subtitle: str
    image: FrameArray


class VideoFrameExtractor:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {video_path}")

        self.fps = _positive_capture_float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(max(self.capture.get(cv2.CAP_PROP_FRAME_COUNT), 0))
        self.width = int(max(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), 0))
        self.height = int(max(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 0))

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoFrameExtractor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read_at_timestamp_ms(self, timestamp_ms: int) -> ExtractedFrame:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative.")

        frame: FrameArray | None = None
        frame_index: int | None = None

        if self.fps is not None and self.fps > 0:
            target_frame_index = int(round((timestamp_ms / 1000.0) * self.fps))
            if self.frame_count > 0:
                target_frame_index = min(max(target_frame_index, 0), self.frame_count - 1)
            for candidate_frame_index in _frame_seek_candidates(target_frame_index):
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_frame_index)
                ok, candidate = self.capture.read()
                if ok and candidate is not None:
                    frame = candidate
                    frame_index = candidate_frame_index
                    break

        if frame is None:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, candidate = self.capture.read()
            if ok and candidate is not None:
                frame = candidate
                capture_position = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                frame_index = max(capture_position, 0)

        if frame is None:
            raise RuntimeError(
                f"Failed to extract frame at {timestamp_ms} ms from {self.video_path}."
            )

        return ExtractedFrame(
            frame=frame,
            frame_index=frame_index,
            source_fps=self.fps,
        )

    def read_at_frame_index(self, frame_index: int) -> ExtractedFrame | None:
        if frame_index < 0:
            return None
        if self.frame_count > 0 and frame_index >= self.frame_count:
            return None

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        capture_position = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        return ExtractedFrame(
            frame=frame,
            frame_index=max(capture_position, 0),
            source_fps=self.fps,
        )


def discover_labeled_video_sources(input_root: Path) -> list[LabeledVideoSource]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    sources: list[LabeledVideoSource] = []
    for csv_path in sorted(input_root.rglob("*.csv")):
        video_path = _resolve_video_path_for_csv(csv_path)
        sources.append(
            LabeledVideoSource(
                video_stem=csv_path.stem,
                video_path=video_path,
                csv_path=csv_path,
            )
        )

    if not sources:
        raise FileNotFoundError(f"No labeled video/CSV pairs were found under {input_root}.")
    return sources


def parse_ground_truth_rows(csv_path: Path, *, video_stem: str) -> list[GroundTruthRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            parse_ground_truth_row(row, row_index=index, video_stem=video_stem)
            for index, row in enumerate(reader, start=1)
        ]


def parse_ground_truth_row(
    row: dict[str, str],
    *,
    row_index: int,
    video_stem: str,
) -> GroundTruthRow:
    bbox = FloatBoundingBox(
        x_min=_parse_decimal(row.get("x_min")),
        y_min=_parse_decimal(row.get("y_min")),
        x_max=_parse_decimal(row.get("x_max")),
        y_max=_parse_decimal(row.get("y_max")),
    )
    if bbox.x_max <= bbox.x_min or bbox.y_max <= bbox.y_min:
        raise ValueError(
            f"Row {row_index} in {video_stem} has a degenerate bounding box: {bbox}."
        )

    return GroundTruthRow(
        row_index=row_index,
        video_stem=video_stem,
        timestamp_ms=normalize_timestamp_ms(row.get("frame_timestamp")),
        bbox=bbox,
        id_sku=_normalize_optional_text(row.get("id_sku")),
        barcode=_normalize_optional_text(row.get("barcode")),
        product_name=_normalize_optional_text(row.get("product_name")),
    )


def group_rows_by_timestamp(rows: list[GroundTruthRow]) -> list[TimestampAnnotationGroup]:
    grouped: dict[int, list[GroundTruthRow]] = defaultdict(list)
    for row in rows:
        grouped[row.timestamp_ms].append(row)

    return [
        TimestampAnnotationGroup(
            video_stem=rows_for_timestamp[0].video_stem,
            timestamp_ms=timestamp_ms,
            rows=tuple(rows_for_timestamp),
        )
        for timestamp_ms, rows_for_timestamp in sorted(grouped.items(), key=lambda item: item[0])
    ]


def resolve_video_split(
    video_stems: list[str],
    *,
    train_videos: list[str] | None = None,
    val_videos: list[str] | None = None,
) -> dict[str, list[str]]:
    known = sorted(dict.fromkeys(video_stems))
    known_set = set(known)

    if not known:
        raise ValueError("At least one video must be provided to build a split.")

    train_override = list(dict.fromkeys(train_videos or []))
    val_override = list(dict.fromkeys(val_videos or []))

    if not train_override and not val_override:
        if {"25_12-20", "26_12-20", "43_15"}.issubset(known_set):
            val = ["43_15"]
            train = [video_stem for video_stem in known if video_stem != "43_15"]
        else:
            if len(known) < 2:
                raise ValueError(
                    "Automatic split requires at least two videos. "
                    "Use --train-videos/--val-videos to override."
                )
            val = [known[-1]]
            train = known[:-1]
        return {"train": train, "val": val}

    _validate_known_video_names(train_override, known_set, field_name="train_videos")
    _validate_known_video_names(val_override, known_set, field_name="val_videos")

    if not train_override:
        train_override = [video_stem for video_stem in known if video_stem not in set(val_override)]
    if not val_override:
        val_override = [video_stem for video_stem in known if video_stem not in set(train_override)]

    train_set = set(train_override)
    val_set = set(val_override)
    if not train_set:
        raise ValueError("Train split cannot be empty.")
    if not val_set:
        raise ValueError("Validation split cannot be empty.")
    if train_set & val_set:
        raise ValueError("Train and val splits must be disjoint at the video level.")
    if train_set | val_set != known_set:
        missing = sorted(known_set - (train_set | val_set))
        raise ValueError(f"Every discovered video must be assigned to a split. Missing: {missing}")

    return {"train": train_override, "val": val_override}


def normalize_timestamp_ms(raw_value: Any) -> int:
    decimal_value = _parse_decimal(raw_value)
    if decimal_value < 0:
        raise ValueError(f"Timestamp must be non-negative: {raw_value!r}")
    return int(
        Decimal(str(decimal_value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def clip_bbox_to_frame(
    bbox: FloatBoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> FloatBoundingBox | None:
    clipped = FloatBoundingBox(
        x_min=max(min(bbox.x_min, float(frame_width)), 0.0),
        y_min=max(min(bbox.y_min, float(frame_height)), 0.0),
        x_max=max(min(bbox.x_max, float(frame_width)), 0.0),
        y_max=max(min(bbox.y_max, float(frame_height)), 0.0),
    )
    if clipped.x_max <= clipped.x_min or clipped.y_max <= clipped.y_min:
        return None
    return clipped


def rotate_bbox(
    bbox: FloatBoundingBox,
    *,
    mode: OrientationMode,
    source_width: int,
    source_height: int,
) -> FloatBoundingBox:
    corners = (
        (bbox.x_min, bbox.y_min),
        (bbox.x_max, bbox.y_min),
        (bbox.x_max, bbox.y_max),
        (bbox.x_min, bbox.y_max),
    )
    if mode == "none":
        transformed = corners
    elif mode == "rotate_90_cw":
        transformed = tuple((source_height - y, x) for x, y in corners)
    elif mode == "rotate_90_ccw":
        transformed = tuple((y, source_width - x) for x, y in corners)
    elif mode == "rotate_180":
        transformed = tuple((source_width - x, source_height - y) for x, y in corners)
    else:
        raise ValueError(f"Unsupported orientation mode: {mode}")

    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return FloatBoundingBox(
        x_min=min(xs),
        y_min=min(ys),
        x_max=max(xs),
        y_max=max(ys),
    )


def bbox_to_yolo(
    bbox: FloatBoundingBox,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    width = bbox.width / float(image_width)
    height = bbox.height / float(image_height)
    x_center = ((bbox.x_min + bbox.x_max) / 2.0) / float(image_width)
    y_center = ((bbox.y_min + bbox.y_max) / 2.0) / float(image_height)

    values = (x_center, y_center, width, height)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"YOLO bbox must stay within [0, 1]: {values}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"YOLO bbox must have positive area: {values}")
    return values


def format_dataset_basename(
    video_stem: str,
    timestamp_ms: int,
    *,
    frame_index: int | None = None,
    frame_offset: int = 0,
) -> str:
    if frame_offset == 0 and frame_index is None:
        return f"{video_stem}_ts_{timestamp_ms}"
    frame_part = "unknown" if frame_index is None else f"{frame_index:06d}"
    return f"{video_stem}_ts_{timestamp_ms}_f_{frame_part}_off_{frame_offset:+04d}"


def extract_frame_at_timestamp(video_path: Path, timestamp_ms: int) -> ExtractedFrame:
    with VideoFrameExtractor(video_path) as extractor:
        return extractor.read_at_timestamp_ms(timestamp_ms)


def build_yolo_price_tag_dataset(
    *,
    input_root: Path,
    output_dir: Path,
    orientation_mode: OrientationMode,
    train_videos: list[str] | None = None,
    val_videos: list[str] | None = None,
    make_orientation_qa: bool = False,
    orientation_review_note: str | None = None,
    dataset_path_value: str | None = None,
    dry_run: bool = False,
    orientation_qa_max_per_video: int = 2,
    propagate_frames: int = 0,
    propagation_ncc_threshold: float = 0.42,
    propagation_frame_stride: int = 1,
    keyframe_holdout_stride: int = 0,
    undistort: bool = False,
) -> dict[str, Any]:
    camera = CameraModel() if undistort else None
    propagate_frames = max(int(propagate_frames), 0)
    propagation_ncc_threshold = float(propagation_ncc_threshold)
    propagation_frame_stride = max(int(propagation_frame_stride), 1)
    keyframe_holdout_stride = max(int(keyframe_holdout_stride), 0)
    sources = discover_labeled_video_sources(input_root)
    split = resolve_video_split(
        [source.video_stem for source in sources],
        train_videos=train_videos,
        val_videos=val_videos,
    )
    split_by_video = {
        video_stem: split_name
        for split_name, video_stems in split.items()
        for video_stem in video_stems
    }

    layout = _DatasetLayout.from_output_dir(output_dir)
    if not dry_run:
        layout.ensure_directories(make_orientation_qa=make_orientation_qa)

    overlay_items_by_split: dict[str, list[ContactSheetItem]] = {"train": [], "val": []}
    crop_items: list[ContactSheetItem] = []
    orientation_qa_selection = _select_orientation_qa_targets(
        sources=sources,
        max_per_video=orientation_qa_max_per_video,
    )
    orientation_qa_counts = {mode: 0 for mode in SUPPORTED_ORIENTATION_MODES}
    boxes_per_video: dict[str, int] = defaultdict(int)
    images_per_video: dict[str, int] = defaultdict(int)
    bbox_areas: list[float] = []
    bbox_aspect_ratios: list[float] = []
    warnings: list[str] = []
    total_rows = 0
    valid_boxes = 0
    skipped_boxes = 0
    images_count = 0
    train_images = 0
    val_images = 0
    keyframe_images = 0
    propagated_images = 0
    propagated_boxes = 0
    propagation_attempts = 0
    keyframe_holdout_by_video: dict[str, list[int]] = {}

    for source in sources:
        split_name = split_by_video[source.video_stem]
        rows = parse_ground_truth_rows(source.csv_path, video_stem=source.video_stem)
        groups = group_rows_by_timestamp(rows)
        total_rows += len(rows)
        holdout_timestamps = _select_holdout_timestamps(
            groups,
            stride=keyframe_holdout_stride if split_name == "train" else 0,
        )
        keyframe_holdout_by_video[source.video_stem] = sorted(holdout_timestamps)

        with VideoFrameExtractor(source.video_path) as extractor:
            source_width = extractor.width
            source_height = extractor.height
            reserved_frame_indices = _timestamps_to_frame_indices(
                holdout_timestamps,
                fps=extractor.fps,
                frame_count=extractor.frame_count,
            )

            for group in groups:
                group_split_name = (
                    "val"
                    if group.timestamp_ms in holdout_timestamps
                    else split_name
                )
                extracted = extractor.read_at_timestamp_ms(group.timestamp_ms)
                oriented_frame = _transform_frame(
                    extracted.frame,
                    orientation_mode=orientation_mode,
                    camera=camera,
                )
                frame_height, frame_width = oriented_frame.shape[:2]
                valid_annotations, invalid_count, annotation_warnings = (
                    _build_valid_annotations_for_frame(
                        rows=group.rows,
                        orientation_mode=orientation_mode,
                        source_width=source_width,
                        source_height=source_height,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        camera=camera,
                    )
                )
                skipped_boxes += invalid_count
                warnings.extend(annotation_warnings)

                if not valid_annotations:
                    continue

                valid_boxes += len(valid_annotations)
                keyframe_images += 1
                boxes_per_video[source.video_stem] += len(valid_annotations)
                bbox_areas.extend(annotation.bbox.area for annotation in valid_annotations)
                bbox_aspect_ratios.extend(
                    annotation.bbox.aspect_ratio for annotation in valid_annotations
                )

                basename = format_dataset_basename(source.video_stem, group.timestamp_ms)
                _record_dataset_image(
                    frame=oriented_frame,
                    group=group,
                    annotations=valid_annotations,
                    basename=basename,
                    split_name=group_split_name,
                    orientation_mode=orientation_mode,
                    layout=layout,
                    dry_run=dry_run,
                    overlay_items_by_split=overlay_items_by_split,
                    crop_items=crop_items,
                )
                images_count += 1
                images_per_video[source.video_stem] += 1
                if group_split_name == "train":
                    train_images += 1
                else:
                    val_images += 1

                if (
                    group_split_name == "train"
                    and propagate_frames > 0
                    and extracted.frame_index is not None
                ):
                    propagated = _propagate_group_annotations(
                        extractor=extractor,
                        reference_frame=oriented_frame,
                        reference_frame_index=extracted.frame_index,
                        group=group,
                        reference_annotations=valid_annotations,
                        orientation_mode=orientation_mode,
                        ncc_threshold=propagation_ncc_threshold,
                        max_frame_offset=propagate_frames,
                        frame_stride=propagation_frame_stride,
                        reserved_frame_indices=reserved_frame_indices,
                        reserved_frame_radius=propagate_frames,
                        camera=camera,
                    )
                    propagation_attempts += len(valid_annotations) * len(
                        _propagation_offsets(
                            propagate_frames,
                            frame_stride=propagation_frame_stride,
                        )
                    )
                    for frame_offset, target_frame_index, propagated_frame, propagated_annotations in propagated:
                        propagated_images += 1
                        propagated_boxes += len(propagated_annotations)
                        valid_boxes += len(propagated_annotations)
                        boxes_per_video[source.video_stem] += len(propagated_annotations)
                        bbox_areas.extend(
                            annotation.bbox.area for annotation in propagated_annotations
                        )
                        bbox_aspect_ratios.extend(
                            annotation.bbox.aspect_ratio
                            for annotation in propagated_annotations
                        )
                        propagated_basename = format_dataset_basename(
                            source.video_stem,
                            group.timestamp_ms,
                            frame_index=target_frame_index,
                            frame_offset=frame_offset,
                        )
                        _record_dataset_image(
                            frame=propagated_frame,
                            group=group,
                            annotations=propagated_annotations,
                            basename=propagated_basename,
                            split_name=group_split_name,
                            orientation_mode=orientation_mode,
                            layout=layout,
                            dry_run=dry_run,
                            overlay_items_by_split=overlay_items_by_split,
                            crop_items=crop_items,
                        )
                        images_count += 1
                        images_per_video[source.video_stem] += 1
                        if group_split_name == "train":
                            train_images += 1
                        else:
                            val_images += 1

                if make_orientation_qa and group.timestamp_ms in orientation_qa_selection[source.video_stem]:
                    for qa_mode in SUPPORTED_ORIENTATION_MODES:
                        qa_overlay = _build_orientation_qa_overlay(
                            frame=extracted.frame,
                            group=group,
                            source_width=source_width,
                            source_height=source_height,
                            orientation_mode=qa_mode,
                        )
                        orientation_qa_counts[qa_mode] += 1
                        if not dry_run:
                            _write_jpeg(
                                layout.orientation_check_dir / qa_mode / f"{basename}.jpg",
                                qa_overlay,
                            )

    dataset_path = dataset_path_value or output_dir.as_posix()
    if not dry_run:
        _write_dataset_yaml(output_dir=output_dir, dataset_path=dataset_path)
        _write_contact_sheet(
            layout.qa_contact_sheets_dir / "train_gt_overlays.jpg",
            _sample_evenly(overlay_items_by_split["train"], limit=12),
            thumb_width=260,
            thumb_height=420,
        )
        _write_contact_sheet(
            layout.qa_contact_sheets_dir / "val_gt_overlays.jpg",
            _sample_evenly(overlay_items_by_split["val"], limit=12),
            thumb_width=260,
            thumb_height=420,
        )
        _write_contact_sheet(
            layout.qa_contact_sheets_dir / "crops_gt_examples.jpg",
            _sample_evenly(crop_items, limit=24),
            thumb_width=260,
            thumb_height=160,
        )

    report = build_yolo_price_tag_dataset_report(
        input_root=input_root,
        output_dir=output_dir,
        dataset_path=dataset_path,
        split=split,
        orientation_mode=orientation_mode,
        orientation_review_note=orientation_review_note,
        make_orientation_qa=make_orientation_qa,
        orientation_qa_counts=orientation_qa_counts,
        total_rows=total_rows,
        valid_boxes=valid_boxes,
        skipped_boxes=skipped_boxes,
        images_count=images_count,
        train_images=train_images,
        val_images=val_images,
        boxes_per_video=boxes_per_video,
        images_per_video=images_per_video,
        bbox_areas=bbox_areas,
        bbox_aspect_ratios=bbox_aspect_ratios,
        keyframe_images=keyframe_images,
        propagated_images=propagated_images,
        propagated_boxes=propagated_boxes,
        propagation_attempts=propagation_attempts,
        propagate_frames=propagate_frames,
        propagation_ncc_threshold=propagation_ncc_threshold,
        propagation_frame_stride=propagation_frame_stride,
        keyframe_holdout_stride=keyframe_holdout_stride,
        keyframe_holdout_by_video=keyframe_holdout_by_video,
        warnings=warnings,
    )
    if not dry_run:
        layout.dataset_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _build_valid_annotations_for_frame(
    *,
    rows: tuple[GroundTruthRow, ...],
    orientation_mode: OrientationMode,
    source_width: int,
    source_height: int,
    frame_width: int,
    frame_height: int,
    camera: CameraModel | None = None,
) -> tuple[list[ValidAnnotation], int, list[str]]:
    annotations: list[ValidAnnotation] = []
    skipped_count = 0
    warnings: list[str] = []

    for row in rows:
        rotated_bbox = _transform_bbox(
            row.bbox,
            orientation_mode=orientation_mode,
            source_width=source_width,
            source_height=source_height,
            camera=camera,
        )
        clipped_bbox = clip_bbox_to_frame(
            rotated_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if clipped_bbox is None:
            skipped_count += 1
            warnings.append(
                f"{row.video_stem} ts={row.timestamp_ms} row={row.row_index}: "
                "bbox was outside the oriented frame after clipping."
            )
            continue

        try:
            yolo_bbox = bbox_to_yolo(
                clipped_bbox,
                image_width=frame_width,
                image_height=frame_height,
            )
        except ValueError as error:
            skipped_count += 1
            warnings.append(
                f"{row.video_stem} ts={row.timestamp_ms} row={row.row_index}: {error}"
            )
            continue

        annotations.append(
            ValidAnnotation(
                row=row,
                bbox=clipped_bbox,
                yolo=yolo_bbox,
            )
        )

    return annotations, skipped_count, warnings


def _record_dataset_image(
    *,
    frame: FrameArray,
    group: TimestampAnnotationGroup,
    annotations: list[ValidAnnotation],
    basename: str,
    split_name: str,
    orientation_mode: OrientationMode,
    layout: "_DatasetLayout",
    dry_run: bool,
    overlay_items_by_split: dict[str, list[ContactSheetItem]],
    crop_items: list[ContactSheetItem],
) -> None:
    overlay = _draw_ground_truth_overlay(
        frame=frame,
        group=group,
        annotations=annotations,
        orientation_mode=orientation_mode,
    )
    overlay_items_by_split[split_name].append(
        ContactSheetItem(
            title=basename,
            subtitle=f"{len(annotations)} box(es)",
            image=overlay,
        )
    )
    crop_items.extend(
        _build_crop_contact_items(
            frame=frame,
            basename=basename,
            annotations=annotations,
        )
    )

    if dry_run:
        return

    _write_jpeg(layout.images_dir / split_name / f"{basename}.jpg", frame)
    _write_label_file(layout.labels_dir / split_name / f"{basename}.txt", annotations)
    _write_jpeg(layout.qa_overlays_dir / f"{basename}.jpg", overlay)


def _propagate_group_annotations(
    *,
    extractor: VideoFrameExtractor,
    reference_frame: FrameArray,
    reference_frame_index: int,
    group: TimestampAnnotationGroup,
    reference_annotations: list[ValidAnnotation],
    orientation_mode: OrientationMode,
    ncc_threshold: float,
    max_frame_offset: int,
    frame_stride: int,
    reserved_frame_indices: set[int],
    reserved_frame_radius: int,
    camera: CameraModel | None = None,
) -> list[tuple[int, int, FrameArray, list[ValidAnnotation]]]:
    if max_frame_offset <= 0:
        return []

    propagated: list[tuple[int, int, FrameArray, list[ValidAnnotation]]] = []
    for frame_offset in _propagation_offsets(
        max_frame_offset,
        frame_stride=frame_stride,
    ):
        target_frame_index = reference_frame_index + frame_offset
        if _is_reserved_propagation_target(
            target_frame_index,
            reserved_frame_indices=reserved_frame_indices,
            reserved_frame_radius=reserved_frame_radius,
        ):
            continue
        extracted = extractor.read_at_frame_index(target_frame_index)
        if extracted is None or extracted.frame_index is None:
            continue

        target_frame = _transform_frame(
            extracted.frame,
            orientation_mode=orientation_mode,
            camera=camera,
        )
        target_height, target_width = target_frame.shape[:2]
        annotations: list[ValidAnnotation] = []
        for reference_annotation in reference_annotations:
            propagated_bbox = _match_template_bbox(
                reference_frame=reference_frame,
                target_frame=target_frame,
                bbox=reference_annotation.bbox,
                ncc_threshold=ncc_threshold,
            )
            if propagated_bbox is None:
                continue
            clipped_bbox = clip_bbox_to_frame(
                propagated_bbox,
                frame_width=target_width,
                frame_height=target_height,
            )
            if clipped_bbox is None:
                continue
            try:
                yolo_bbox = bbox_to_yolo(
                    clipped_bbox,
                    image_width=target_width,
                    image_height=target_height,
                )
            except ValueError:
                continue
            annotations.append(
                ValidAnnotation(
                    row=reference_annotation.row,
                    bbox=clipped_bbox,
                    yolo=yolo_bbox,
                )
            )

        if annotations:
            propagated.append(
                (
                    frame_offset,
                    extracted.frame_index,
                    target_frame,
                    annotations,
                )
            )

    return propagated


def _propagation_offsets(max_frame_offset: int, *, frame_stride: int = 1) -> list[int]:
    offsets: list[int] = []
    for offset in range(frame_stride, max_frame_offset + 1, frame_stride):
        offsets.append(-offset)
        offsets.append(offset)
    return offsets


def _select_holdout_timestamps(
    groups: list[TimestampAnnotationGroup],
    *,
    stride: int,
) -> set[int]:
    if stride <= 0 or len(groups) < max(stride, 3):
        return set()
    holdout = {
        group.timestamp_ms
        for index, group in enumerate(groups, start=1)
        if index % stride == 0
    }
    if len(holdout) >= len(groups):
        holdout = {groups[-1].timestamp_ms}
    return holdout


def _timestamps_to_frame_indices(
    timestamps_ms: set[int],
    *,
    fps: float | None,
    frame_count: int,
) -> set[int]:
    if not timestamps_ms or fps is None or fps <= 0:
        return set()
    indices: set[int] = set()
    for timestamp_ms in timestamps_ms:
        frame_index = int(round((timestamp_ms / 1000.0) * fps))
        if frame_count > 0:
            frame_index = min(max(frame_index, 0), frame_count - 1)
        indices.add(frame_index)
    return indices


def _is_reserved_propagation_target(
    target_frame_index: int,
    *,
    reserved_frame_indices: set[int],
    reserved_frame_radius: int,
) -> bool:
    if not reserved_frame_indices:
        return False
    return any(
        abs(target_frame_index - reserved_frame_index) <= reserved_frame_radius
        for reserved_frame_index in reserved_frame_indices
    )


def _match_template_bbox(
    *,
    reference_frame: FrameArray,
    target_frame: FrameArray,
    bbox: FloatBoundingBox,
    ncc_threshold: float,
) -> FloatBoundingBox | None:
    ref_height, ref_width = reference_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    x_min, y_min, x_max, y_max = _bbox_to_draw_coords(
        bbox,
        frame_width=ref_width,
        frame_height=ref_height,
    )
    template = reference_frame[y_min:y_max, x_min:x_max]
    if template.size == 0:
        return None
    template_height, template_width = template.shape[:2]
    if template_width > target_width or template_height > target_height:
        return None

    reference_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    search_x_min, search_y_min, search_x_max, search_y_max = _template_search_window(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        frame_width=target_width,
        frame_height=target_height,
    )
    search_region = target_gray[search_y_min:search_y_max, search_x_min:search_x_max]
    if search_region.shape[0] < template_height or search_region.shape[1] < template_width:
        return None

    result = cv2.matchTemplate(search_region, reference_gray, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    if max_value < ncc_threshold:
        return None

    match_x = search_x_min + max_location[0]
    match_y = search_y_min + max_location[1]
    return FloatBoundingBox(
        x_min=float(match_x),
        y_min=float(match_y),
        x_max=float(match_x + template_width),
        y_max=float(match_y + template_height),
    )


def _template_search_window(
    *,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    margin_x = max(int(round(width * 2.5)), 160)
    margin_y = max(int(round(height * 2.5)), 160)
    return (
        max(x_min - margin_x, 0),
        max(y_min - margin_y, 0),
        min(x_max + margin_x, frame_width),
        min(y_max + margin_y, frame_height),
    )


def build_yolo_price_tag_dataset_report(
    *,
    input_root: Path,
    output_dir: Path,
    dataset_path: str,
    split: dict[str, list[str]],
    orientation_mode: OrientationMode,
    orientation_review_note: str | None,
    make_orientation_qa: bool,
    orientation_qa_counts: dict[str, int],
    total_rows: int,
    valid_boxes: int,
    skipped_boxes: int,
    images_count: int,
    train_images: int,
    val_images: int,
    boxes_per_video: dict[str, int],
    images_per_video: dict[str, int],
    bbox_areas: list[float],
    bbox_aspect_ratios: list[float],
    keyframe_images: int,
    propagated_images: int,
    propagated_boxes: int,
    propagation_attempts: int,
    propagate_frames: int,
    propagation_ncc_threshold: float,
    propagation_frame_stride: int,
    keyframe_holdout_stride: int,
    keyframe_holdout_by_video: dict[str, list[int]],
    warnings: list[str],
) -> dict[str, Any]:
    orientation_review = orientation_review_note
    if orientation_review is None:
        if make_orientation_qa:
            orientation_review = (
                "Orientation QA overlays were generated. "
                f"The dataset build used '{orientation_mode}'. "
                "Open qa/orientation_check/* to confirm that this mode looks correct."
            )
        else:
            orientation_review = (
                f"The dataset build used '{orientation_mode}', but orientation QA overlays "
                "were not generated in this run."
            )

    return {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "dataset_yaml_path": str(output_dir / "dataset.yaml"),
        "dataset_path": dataset_path,
        "class_names": {str(CLASS_ID): CLASS_NAME},
        "train_videos": list(split["train"]),
        "val_videos": list(split["val"]),
        "total_rows": total_rows,
        "valid_boxes": valid_boxes,
        "skipped_boxes": skipped_boxes,
        "images_count": images_count,
        "train_images": train_images,
        "val_images": val_images,
        "keyframe_images": keyframe_images,
        "propagated_images": propagated_images,
        "propagated_boxes": propagated_boxes,
        "propagation_attempts": propagation_attempts,
        "propagate_frames": propagate_frames,
        "propagation_ncc_threshold": propagation_ncc_threshold,
        "propagation_frame_stride": propagation_frame_stride,
        "keyframe_holdout_stride": keyframe_holdout_stride,
        "keyframe_holdout_by_video": {
            video_stem: timestamps
            for video_stem, timestamps in sorted(keyframe_holdout_by_video.items())
            if timestamps
        },
        "boxes_per_video": dict(sorted(boxes_per_video.items())),
        "images_per_video": dict(sorted(images_per_video.items())),
        "bbox_area": _summarize_numeric_series(bbox_areas),
        "bbox_aspect_ratio": _summarize_numeric_series(bbox_aspect_ratios),
        "chosen_orientation_mode": orientation_mode,
        "orientation_review": orientation_review,
        "orientation_qa_generated": make_orientation_qa,
        "orientation_qa_counts": orientation_qa_counts,
        "qa_paths": {
            "overlays_dir": str(output_dir / "qa" / "overlays"),
            "orientation_check_dir": str(output_dir / "qa" / "orientation_check"),
            "contact_sheets_dir": str(output_dir / "qa" / "contact_sheets"),
            "dataset_report_path": str(output_dir / "qa" / "dataset_report.json"),
        },
        "warnings": warnings,
    }


@dataclass(frozen=True, slots=True)
class _DatasetLayout:
    output_dir: Path
    images_dir: Path
    labels_dir: Path
    qa_dir: Path
    qa_overlays_dir: Path
    qa_contact_sheets_dir: Path
    orientation_check_dir: Path
    dataset_report_path: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "_DatasetLayout":
        qa_dir = output_dir / "qa"
        return cls(
            output_dir=output_dir,
            images_dir=output_dir / "images",
            labels_dir=output_dir / "labels",
            qa_dir=qa_dir,
            qa_overlays_dir=qa_dir / "overlays",
            qa_contact_sheets_dir=qa_dir / "contact_sheets",
            orientation_check_dir=qa_dir / "orientation_check",
            dataset_report_path=qa_dir / "dataset_report.json",
        )

    def ensure_directories(self, *, make_orientation_qa: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val"):
            (self.images_dir / split_name).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split_name).mkdir(parents=True, exist_ok=True)
        self.qa_overlays_dir.mkdir(parents=True, exist_ok=True)
        self.qa_contact_sheets_dir.mkdir(parents=True, exist_ok=True)
        if make_orientation_qa:
            for orientation_mode in SUPPORTED_ORIENTATION_MODES:
                (self.orientation_check_dir / orientation_mode).mkdir(parents=True, exist_ok=True)


def _resolve_video_path_for_csv(csv_path: Path) -> Path:
    exact_match = csv_path.with_suffix(".mp4")
    if exact_match.exists():
        return exact_match

    sibling_videos = sorted(csv_path.parent.glob("*.mp4"))
    if len(sibling_videos) == 1:
        return sibling_videos[0]
    if not sibling_videos:
        raise FileNotFoundError(f"No MP4 file found for labeled CSV: {csv_path}")
    raise RuntimeError(f"Multiple MP4 files found for labeled CSV: {csv_path}")


def _validate_known_video_names(
    video_names: list[str],
    known_video_names: set[str],
    *,
    field_name: str,
) -> None:
    unknown = sorted(set(video_names) - known_video_names)
    if unknown:
        raise ValueError(f"Unknown {field_name}: {unknown}")


def _parse_decimal(raw_value: Any) -> float:
    if raw_value is None:
        raise ValueError("Required numeric value is missing.")
    raw_text = str(raw_value).strip().replace(" ", "").replace(",", ".")
    if not raw_text:
        raise ValueError("Required numeric value is empty.")
    try:
        return float(Decimal(raw_text))
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric value: {raw_value!r}") from error


def _normalize_optional_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text.lower() in PLACEHOLDER_VALUES:
        return None
    return text or None


def _positive_capture_float(value: float) -> float | None:
    if value <= 0:
        return None
    return float(value)


def _frame_seek_candidates(target_frame_index: int, *, max_backtrack: int = 60) -> list[int]:
    candidates = [max(target_frame_index, 0)]
    for offset in range(1, max_backtrack + 1):
        candidate = target_frame_index - offset
        if candidate < 0:
            break
        candidates.append(candidate)
    return candidates


def _draw_ground_truth_overlay(
    *,
    frame: FrameArray,
    group: TimestampAnnotationGroup,
    annotations: list[ValidAnnotation],
    orientation_mode: OrientationMode,
) -> FrameArray:
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    font_scale = max(min(frame_width, frame_height) / 2400.0, 0.5)
    thickness = max(int(round(font_scale * 2)), 1)

    header = (
        f"{group.video_stem} ts={group.timestamp_ms}ms "
        f"boxes={len(annotations)} orientation={orientation_mode}"
    )
    _draw_text_box(
        annotated,
        header,
        origin=(18, 26),
        font_scale=font_scale,
        thickness=thickness,
    )

    for index, annotation in enumerate(annotations, start=1):
        x_min, y_min, x_max, y_max = _bbox_to_draw_coords(
            annotation.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        cv2.rectangle(
            annotated,
            (x_min, y_min),
            (x_max, y_max),
            OVERLAY_BOX_COLOR,
            thickness,
        )
        label = f"#{index} {_annotation_short_id(annotation.row)}"
        label_y = max(y_min - 10, 24)
        _draw_text_box(
            annotated,
            label,
            origin=(x_min, label_y),
            font_scale=font_scale,
            thickness=thickness,
        )

    return annotated


def _build_orientation_qa_overlay(
    *,
    frame: FrameArray,
    group: TimestampAnnotationGroup,
    source_width: int,
    source_height: int,
    orientation_mode: OrientationMode,
) -> FrameArray:
    oriented = apply_orientation(frame, orientation_mode)
    frame_height, frame_width = oriented.shape[:2]
    annotations: list[ValidAnnotation] = []

    for row in group.rows:
        rotated_bbox = rotate_bbox(
            row.bbox,
            mode=orientation_mode,
            source_width=source_width,
            source_height=source_height,
        )
        clipped_bbox = clip_bbox_to_frame(
            rotated_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if clipped_bbox is None:
            continue
        annotations.append(
            ValidAnnotation(
                row=row,
                bbox=clipped_bbox,
                yolo=(0.0, 0.0, 0.0, 0.0),
            )
        )

    return _draw_ground_truth_overlay(
        frame=oriented,
        group=group,
        annotations=annotations,
        orientation_mode=orientation_mode,
    )


def _build_crop_contact_items(
    *,
    frame: FrameArray,
    basename: str,
    annotations: list[ValidAnnotation],
) -> list[ContactSheetItem]:
    frame_height, frame_width = frame.shape[:2]
    items: list[ContactSheetItem] = []

    for index, annotation in enumerate(annotations, start=1):
        x_min, y_min, x_max, y_max = _bbox_to_draw_coords(
            annotation.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        crop = frame[y_min:y_max, x_min:x_max].copy()
        if crop.size == 0:
            continue
        items.append(
            ContactSheetItem(
                title=f"{basename} #{index}",
                subtitle=f"{crop.shape[1]}x{crop.shape[0]} {_annotation_short_id(annotation.row)}",
                image=crop,
            )
        )
    return items


def _annotation_short_id(row: GroundTruthRow) -> str:
    if row.id_sku:
        return f"sku={row.id_sku}"
    if row.barcode:
        return f"barcode={row.barcode}"
    return f"row={row.row_index}"


def _bbox_to_draw_coords(
    bbox: FloatBoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x_min = max(min(int(math.floor(bbox.x_min)), frame_width - 1), 0)
    y_min = max(min(int(math.floor(bbox.y_min)), frame_height - 1), 0)
    x_max = max(min(int(math.ceil(bbox.x_max)), frame_width), x_min + 1)
    y_max = max(min(int(math.ceil(bbox.y_max)), frame_height), y_min + 1)
    return x_min, y_min, x_max, y_max


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


def _write_label_file(path: Path, annotations: list[ValidAnnotation]) -> None:
    lines = [
        f"{CLASS_ID} {annotation.yolo[0]:.6f} {annotation.yolo[1]:.6f} "
        f"{annotation.yolo[2]:.6f} {annotation.yolo[3]:.6f}"
        for annotation in annotations
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dataset_yaml(*, output_dir: Path, dataset_path: str) -> None:
    payload = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "names": {CLASS_ID: CLASS_NAME},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_jpeg(path: Path, image: FrameArray, *, quality: int = DEFAULT_JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        raise RuntimeError(f"Failed to encode JPEG: {path}")
    path.write_bytes(encoded.tobytes())


def _write_contact_sheet(
    path: Path,
    items: list[ContactSheetItem],
    *,
    thumb_width: int,
    thumb_height: int,
) -> None:
    sheet = _build_contact_sheet_image(
        items=items,
        thumb_width=thumb_width,
        thumb_height=thumb_height,
    )
    _write_jpeg(path, sheet, quality=92)


def _build_contact_sheet_image(
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


def _sample_evenly(items: list[ContactSheetItem], *, limit: int) -> list[ContactSheetItem]:
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[index] for index in indices.tolist()]


def _select_orientation_qa_targets(
    *,
    sources: list[LabeledVideoSource],
    max_per_video: int,
) -> dict[str, set[int]]:
    selection: dict[str, set[int]] = {}
    for source in sources:
        rows = parse_ground_truth_rows(source.csv_path, video_stem=source.video_stem)
        groups = group_rows_by_timestamp(rows)
        if not groups:
            selection[source.video_stem] = set()
            continue
        timestamps = [group.timestamp_ms for group in groups]
        if len(timestamps) <= max_per_video:
            selection[source.video_stem] = set(timestamps)
            continue
        indices = np.linspace(0, len(timestamps) - 1, num=max_per_video, dtype=int)
        selection[source.video_stem] = {timestamps[index] for index in indices.tolist()}
    return selection


def _summarize_numeric_series(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "max": round(max(values), 6),
    }
