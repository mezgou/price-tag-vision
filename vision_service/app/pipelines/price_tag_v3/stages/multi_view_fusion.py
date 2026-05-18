from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome


@dataclass(slots=True)
class _FusionConfig:
    enabled: bool
    max_tracks: int
    max_crops_per_track: int
    min_crops_per_track: int
    canonical_width: int
    canonical_height: int
    upload_debug_images: bool


class MultiViewFusionStage(BaseStage):
    name = "MultiViewFusionStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        tracks = context.artifacts.get("track_crop_bank", [])
        return {
            "enabled": config.enabled,
            "track_bank_count": len(tracks) if isinstance(tracks, list) else 0,
            "max_tracks": config.max_tracks,
            "max_crops_per_track": config.max_crops_per_track,
            "canonical_size": [config.canonical_width, config.canonical_height],
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        raw_tracks = context.artifacts.get("track_crop_bank", [])
        if not config.enabled or not isinstance(raw_tracks, list) or not raw_tracks:
            context.artifacts["v3_fused_tracks"] = []
            return StageOutcome(output_summary={"enabled": config.enabled, "fused_tracks": 0})

        output_dir = context.work_dir / "runtime" / "multi_view_fusion"
        output_dir.mkdir(parents=True, exist_ok=True)
        fused_tracks: list[dict[str, Any]] = []
        skipped_tracks = 0
        warnings: list[str] = []

        for track in raw_tracks[: config.max_tracks]:
            crops = track.get("crops", []) if isinstance(track, dict) else []
            if not isinstance(crops, list) or len(crops) < config.min_crops_per_track:
                skipped_tracks += 1
                continue

            selected = _select_crops(crops, limit=config.max_crops_per_track)
            aligned: list[np.ndarray] = []
            selected_payloads: list[dict[str, Any]] = []
            reference_gray: np.ndarray | None = None
            for crop_payload in selected:
                path = Path(str(crop_payload.get("local_path", "")))
                image = cv2.imread(str(path))
                if image is None:
                    continue
                rectified = _rectify_to_canonical(
                    image,
                    width=config.canonical_width,
                    height=config.canonical_height,
                )
                if reference_gray is None:
                    reference_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
                    aligned.append(rectified)
                else:
                    aligned.append(_register_to_reference(rectified, reference_gray))
                selected_payloads.append(crop_payload)

            if len(aligned) < config.min_crops_per_track:
                skipped_tracks += 1
                continue

            fused = _robust_fuse(aligned)
            deblurred = _deblur_variant(fused)
            track_id = str(track.get("track_id") or f"track_{len(fused_tracks) + 1:05d}")
            fused_path = output_dir / f"{track_id}_fused.jpg"
            deblurred_path = output_dir / f"{track_id}_deblurred.jpg"
            cv2.imwrite(str(fused_path), fused, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            cv2.imwrite(str(deblurred_path), deblurred, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

            fused_key = ""
            deblurred_key = ""
            if config.upload_debug_images:
                fused_key = _upload_image(
                    context,
                    f"debug/v3_fusion/{track_id}_fused.jpg",
                    fused,
                )
                deblurred_key = _upload_image(
                    context,
                    f"debug/v3_fusion/{track_id}_deblurred.jpg",
                    deblurred,
                )

            anchor = max(
                selected_payloads,
                key=lambda item: (
                    float(item.get("quality", {}).get("score", 0.0)),
                    float(item.get("area", 0.0)),
                ),
            )
            fused_tracks.append(
                {
                    "track_id": track_id,
                    "selected_crops": len(selected_payloads),
                    "anchor_crop_id": anchor.get("crop_id", ""),
                    "anchor_detection_id": anchor.get("detection_id", ""),
                    "anchor_frame_index": anchor.get("frame_index", 0),
                    "anchor_timestamp_ms": anchor.get("timestamp_ms"),
                    "anchor_bbox": anchor.get("bbox", {}),
                    "local_path": str(fused_path),
                    "deblurred_local_path": str(deblurred_path),
                    "debug_key": fused_key,
                    "deblurred_debug_key": deblurred_key,
                    "canonical_width": config.canonical_width,
                    "canonical_height": config.canonical_height,
                }
            )

        context.artifacts["v3_fused_tracks"] = fused_tracks
        context.artifacts["v3_fusion_debug_keys"] = [
            item["debug_key"] for item in fused_tracks if item.get("debug_key")
        ]
        if skipped_tracks and not fused_tracks:
            warnings.append("No tracks had enough readable crops for multi-view fusion.")
        return StageOutcome(
            output_summary={
                "enabled": True,
                "fused_tracks": len(fused_tracks),
                "skipped_tracks": skipped_tracks,
                "debug_images": len(context.artifacts["v3_fusion_debug_keys"]),
            },
            warnings=warnings,
        )


def _select_crops(crops: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    def score(item: dict[str, Any]) -> tuple[float, float, float]:
        quality = item.get("quality", {})
        quality_score = float(quality.get("score", 0.0)) if isinstance(quality, dict) else 0.0
        area = float(item.get("area", 0.0))
        glare = float(item.get("glare_ratio", 1.0))
        return (quality_score - 0.25 * glare, area, -glare)

    return sorted(crops, key=score, reverse=True)[:limit]


def _rectify_to_canonical(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    # Detector crops already follow the tag bounding box. The first v3
    # rectifier keeps that geometry but normalizes every observation into a
    # shared coordinate system; contour/QR quadrilateral refinement can replace
    # this function without changing downstream stages.
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _register_to_reference(image: np.ndarray, reference_gray: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        shift, response = cv2.phaseCorrelate(
            np.float32(reference_gray),
            np.float32(gray),
        )
    except Exception:  # noqa: BLE001
        return image
    if not np.isfinite(response) or response < 0.03:
        return image
    dx, dy = shift
    if abs(dx) > 40 or abs(dy) > 40:
        return image
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _robust_fuse(images: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(images, axis=0).astype(np.float32)
    median = np.median(stack, axis=0)
    mean = np.mean(stack, axis=0)
    fused = (0.65 * median) + (0.35 * mean)
    return np.clip(fused, 0, 255).astype(np.uint8)


def _deblur_variant(image: np.ndarray) -> np.ndarray:
    gaussian = cv2.GaussianBlur(image, (0, 0), 1.2)
    sharpened = cv2.addWeighted(image, 1.75, gaussian, -0.75, 0)
    return cv2.bilateralFilter(sharpened, 5, 32, 32)


def _upload_image(context: PipelineContext, relative_path: str, image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 94],
    )
    if not ok:
        raise RuntimeError(f"Failed to encode fusion debug image {relative_path}.")
    return context.artifact_writer.upload_bytes(
        relative_path,
        encoded.tobytes(),
        "image/jpeg",
    )


def _config(context: PipelineContext) -> _FusionConfig:
    raw = context.config.get("multi_view_fusion", {})
    if not isinstance(raw, dict):
        raw = {}
    return _FusionConfig(
        enabled=_bool(raw.get("enabled"), default=True),
        max_tracks=_positive_int(raw.get("max_tracks"), 120),
        max_crops_per_track=_positive_int(raw.get("max_crops_per_track"), 20),
        min_crops_per_track=_positive_int(raw.get("min_crops_per_track"), 2),
        canonical_width=_positive_int(raw.get("canonical_width"), 900),
        canonical_height=_positive_int(raw.get("canonical_height"), 1200),
        upload_debug_images=_bool(raw.get("upload_debug_images"), default=True),
    )


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
