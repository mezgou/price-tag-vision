from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.zonal_ocr import (
    ZonalOcrConfig,
    _create_engine,
    _fields_from_zones,
    _recognize_zonal,
    classify_tag_color,
    parse_ocr_fields,
)
from app.schemas.detections import CropCandidate


@dataclass(slots=True)
class _FusedZonalConfig:
    enabled: bool
    max_tracks: int


class FusedZonalDecodeStage(BaseStage):
    name = "FusedZonalDecodeStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = _config(context)
        fused = context.artifacts.get("v3_fused_tracks", [])
        return {
            "enabled": config.enabled,
            "fused_tracks": len(fused) if isinstance(fused, list) else 0,
            "max_tracks": config.max_tracks,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = _config(context)
        fused_tracks = context.artifacts.get("v3_fused_tracks", [])
        if not config.enabled or not isinstance(fused_tracks, list) or not fused_tracks:
            context.artifacts["fused_zonal_decode"] = {
                "enabled": config.enabled,
                "tracks_processed": 0,
                "tracks_with_fields": 0,
            }
            return StageOutcome(output_summary=context.artifacts["fused_zonal_decode"])

        warnings: list[str] = []
        zonal_config = ZonalOcrConfig.from_context(context)
        warnings.extend(zonal_config.warnings)
        engines = _resolve_engines(config=zonal_config, warnings=warnings)
        crops_by_id = {crop.crop_id: crop for crop in context.crop_candidates}

        tracks_processed = 0
        tracks_with_text = 0
        tracks_with_fields = 0
        field_counts: dict[str, int] = {}

        for fused in fused_tracks[: config.max_tracks]:
            crop = crops_by_id.get(str(fused.get("anchor_crop_id", "")))
            image = _read_best_variant(fused)
            if image is None:
                continue
            tracks_processed += 1

            zonal = _recognize_zonal(image, engines=engines)
            fields = _fields_from_zones(zonal.zone_texts)
            if not fields.get("product_name") or not fields.get("barcode"):
                blob = parse_ocr_fields(zonal.combined_text)
                for key, value in blob.items():
                    fields.setdefault(key, value)
            color = classify_tag_color(image)
            if color:
                fields["color"] = color

            if zonal.combined_text:
                tracks_with_text += 1
            if fields:
                tracks_with_fields += 1
                for field in fields:
                    field_counts[field] = field_counts.get(field, 0) + 1
            if crop is not None:
                _merge_ocr_fields(
                    crop,
                    fields=fields,
                    text=zonal.combined_text,
                    confidence=zonal.confidence,
                    engine=zonal.engine_name,
                )
                crop.attributes["v3_fused_image"] = {
                    "track_id": fused.get("track_id", ""),
                    "debug_key": fused.get("debug_key", ""),
                    "deblurred_debug_key": fused.get("deblurred_debug_key", ""),
                }

        summary = {
            "enabled": True,
            "tracks_processed": tracks_processed,
            "tracks_with_text": tracks_with_text,
            "tracks_with_fields": tracks_with_fields,
            "field_counts": field_counts,
        }
        context.artifacts["fused_zonal_decode"] = summary
        return StageOutcome(output_summary=summary, warnings=warnings)


def _resolve_engines(*, config: ZonalOcrConfig, warnings: list[str]) -> list[Any]:
    engines = []
    for engine_name in config.engine_order:
        engine = _create_engine(engine_name, config=config, warnings=warnings)
        if engine is not None:
            engines.append(engine)
    return engines


def _read_best_variant(fused: dict[str, Any]) -> Any | None:
    for key in ("deblurred_local_path", "local_path"):
        path = Path(str(fused.get(key, "")))
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            return image
    return None


def _merge_ocr_fields(
    crop: CropCandidate,
    *,
    fields: dict[str, str],
    text: str,
    confidence: float,
    engine: str,
) -> None:
    existing = crop.attributes.get("ocr")
    if not isinstance(existing, dict):
        existing = {}
    merged_fields = dict(existing.get("fields", {})) if isinstance(existing.get("fields"), dict) else {}
    for key, value in fields.items():
        if value:
            merged_fields[key] = value
    crop.attributes["ocr"] = {
        **existing,
        "text": "\n".join(item for item in [str(existing.get("text", "")), text] if item),
        "confidence": max(float(existing.get("confidence", 0.0) or 0.0), float(confidence or 0.0)),
        "engine": engine or existing.get("engine", "fused_zonal"),
        "orientation": "fused_zonal",
        "fields": merged_fields,
    }


def _config(context: PipelineContext) -> _FusedZonalConfig:
    raw = context.config.get("fused_zonal_decode", {})
    if not isinstance(raw, dict):
        raw = {}
    return _FusedZonalConfig(
        enabled=_bool(raw.get("enabled"), default=True),
        max_tracks=_positive_int(raw.get("max_tracks"), 120),
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
