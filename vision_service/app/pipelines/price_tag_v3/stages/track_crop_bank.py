from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, SampledFrameMetadata, StageOutcome
from app.schemas.detections import CropCandidate
from app.utils.image_processing import clip_bbox_to_frame


@dataclass(slots=True)
class _TrackCropBankConfig:
    enabled: bool
    max_tracks: int
    min_crops_per_track: int
    save_debug_crops: bool


class TrackCropBankStage(BaseStage):
    name = "TrackCropBankStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_tracks": config.max_tracks,
            "min_crops_per_track": config.min_crops_per_track,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        if not config.enabled or not context.crop_candidates:
            context.artifacts["track_crop_bank"] = []
            return StageOutcome(output_summary={"enabled": config.enabled, "tracks": 0})

        frame_lookup = {frame.frame_index: frame for frame in context.sampled_frames}
        by_track: dict[str, list[CropCandidate]] = defaultdict(list)
        for crop in context.crop_candidates:
            by_track[_track_id(crop)].append(crop)

        bank_dir = context.work_dir / "runtime" / "track_crop_bank"
        bank_dir.mkdir(parents=True, exist_ok=True)
        track_entries: list[dict[str, Any]] = []
        skipped_images = 0

        for track_id, crops in sorted(
            by_track.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[: config.max_tracks]:
            if len(crops) < config.min_crops_per_track:
                continue
            crop_entries: list[dict[str, Any]] = []
            for index, crop in enumerate(
                sorted(crops, key=lambda item: item.quality.score, reverse=True),
                start=1,
            ):
                image = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
                if image is None:
                    skipped_images += 1
                    continue
                local_path = bank_dir / f"{track_id}_{index:03d}_{crop.crop_id}.jpg"
                cv2.imwrite(
                    str(local_path),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                crop_entries.append(
                    {
                        "crop_id": crop.crop_id,
                        "detection_id": crop.detection_id,
                        "track_id": track_id,
                        "frame_index": crop.frame_index,
                        "timestamp_ms": crop.timestamp_ms,
                        "bbox": crop.bbox.model_dump(),
                        "padded_bbox": crop.padded_bbox.model_dump(),
                        "quality": crop.quality.model_dump(),
                        "width": crop.width,
                        "height": crop.height,
                        "local_path": str(local_path),
                        "sharpness": crop.quality.sharpness,
                        "area": crop.width * crop.height,
                        "glare_ratio": crop.quality.glare_ratio,
                        "motion_blur": _estimate_motion_blur(image),
                        "homography": _identity_homography(),
                    }
                )
            if crop_entries:
                track_entries.append(
                    {
                        "track_id": track_id,
                        "crops": crop_entries,
                        "crops_count": len(crop_entries),
                    }
                )

        context.artifacts["track_crop_bank"] = track_entries
        return StageOutcome(
            output_summary={
                "enabled": True,
                "tracks": len(track_entries),
                "bank_crops": sum(len(item["crops"]) for item in track_entries),
                "skipped_images": skipped_images,
            }
        )


def _config(context: PipelineContext) -> _TrackCropBankConfig:
    raw = context.config.get("track_crop_bank", {})
    if not isinstance(raw, dict):
        raw = {}
    return _TrackCropBankConfig(
        enabled=_bool(raw.get("enabled"), default=True),
        max_tracks=_positive_int(raw.get("max_tracks"), 160),
        min_crops_per_track=_positive_int(raw.get("min_crops_per_track"), 1),
        save_debug_crops=_bool(raw.get("save_debug_crops"), default=False),
    )


def _track_id(crop: CropCandidate) -> str:
    value = crop.attributes.get("track_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return crop.detection_id


def _load_crop_image(
    *,
    crop: CropCandidate,
    frame_lookup: dict[int, SampledFrameMetadata],
) -> np.ndarray | None:
    frame_meta = frame_lookup.get(crop.frame_index)
    if frame_meta is None or frame_meta.local_frame_path is None:
        return None
    frame_path = Path(frame_meta.local_frame_path)
    if not frame_path.exists():
        return None
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    padded_bbox = clip_bbox_to_frame(
        crop.padded_bbox,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )
    if padded_bbox is None:
        return None
    image = frame[padded_bbox.y_min : padded_bbox.y_max, padded_bbox.x_min : padded_bbox.x_max]
    if image.size == 0:
        return None
    return image.copy()


def _estimate_motion_blur(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(max(0.0, min(1.0, 1.0 - (lap_var / 500.0))))


def _identity_homography() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
