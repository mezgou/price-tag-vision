from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.schemas.detections import (
    CropCandidate,
    DecodeAttempt,
    DecodedSymbol,
    DetectionCandidate,
)
from app.utils.decoding import build_payload_preview
from app.utils.artifacts import ArtifactWriter
from app.utils.timing import utc_now


@dataclass(slots=True)
class VideoMetadata:
    filename: str
    input_video_key: str
    duration_ms: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_input_key(cls, input_video_key: str) -> "VideoMetadata":
        return cls(
            filename=Path(input_video_key).name,
            input_video_key=input_video_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "input_video_key": self.input_video_key,
            "duration_ms": self.duration_ms,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class StageOutcome:
    output_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StageExecution:
    name: str
    status: str
    duration_ms: int
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(slots=True)
class SampledFrameMetadata:
    frame_index: int
    timestamp_ms: int | None
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    orientation_applied: str
    debug_frame_key: str | None = None
    local_frame_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "orientation_applied": self.orientation_applied,
            "debug_frame_key": self.debug_frame_key,
        }


@dataclass(slots=True)
class PipelineContext:
    job_id: str
    input_video_key: str
    pipeline_name: str
    pipeline_version: str
    config: dict[str, Any]
    local_video_path: Path
    work_dir: Path
    output_dir: Path
    artifact_writer: ArtifactWriter
    video_metadata: VideoMetadata
    artifacts: dict[str, Any] = field(default_factory=dict)
    detections: list[DetectionCandidate] = field(default_factory=list)
    crop_candidates: list[CropCandidate] = field(default_factory=list)
    decode_attempts: list[DecodeAttempt] = field(default_factory=list)
    decoded_symbols: list[DecodedSymbol] = field(default_factory=list)
    csv_rows: list[dict[str, str]] = field(default_factory=list)
    sampled_frames: list[SampledFrameMetadata] = field(default_factory=list)
    debug_frame_keys: list[str] = field(default_factory=list)
    debug_overlay_keys: list[str] = field(default_factory=list)
    debug_crop_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    stage_reports: list[StageExecution] = field(default_factory=list)
    frames_processed: int = 0
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    def build_manifest(
        self,
        *,
        extra_stages: list[StageExecution] | None = None,
        finished_at: datetime | None = None,
    ) -> dict[str, Any]:
        stage_reports = [report.to_dict() for report in self.stage_reports]
        if extra_stages:
            stage_reports.extend(report.to_dict() for report in extra_stages)

        return {
            "job_id": self.job_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": (finished_at or self.finished_at or utc_now()).isoformat(),
            "stages": stage_reports,
            "video_metadata": self.video_metadata.to_dict(),
            "config": self.config,
            "artifacts": self.artifacts,
            "stats": self.build_stats(),
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def detections_by_frame(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            key = str(detection.frame_index)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def serialize_detections(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        detections = self.detections if limit is None else self.detections[:limit]
        return [detection.model_dump() for detection in detections]

    def serialize_crops(
        self,
        *,
        limit: int | None = None,
        sort_by_quality: bool = False,
    ) -> list[dict[str, Any]]:
        crops = list(self.crop_candidates)
        if sort_by_quality:
            crops.sort(key=lambda crop: crop.quality.score, reverse=True)
        if limit is not None:
            crops = crops[:limit]
        return [crop.model_dump() for crop in crops]

    def decoded_symbols_by_type(self) -> dict[str, int]:
        counts = {"qr": 0, "barcode": 0, "unknown": 0}
        for symbol in self.decoded_symbols:
            counts[symbol.symbol_type] = counts.get(symbol.symbol_type, 0) + 1
        return counts

    def serialize_decoded_symbols_preview(
        self,
        *,
        limit: int | None = None,
        max_payload_preview_length: int,
    ) -> list[dict[str, Any]]:
        symbols = self.decoded_symbols if limit is None else self.decoded_symbols[:limit]
        return [
            {
                "symbol_id": symbol.symbol_id,
                "crop_id": symbol.crop_id,
                "detection_id": symbol.detection_id,
                "frame_index": symbol.frame_index,
                "timestamp_ms": symbol.timestamp_ms,
                "symbol_type": symbol.symbol_type,
                "decoder": symbol.decoder,
                "variant": symbol.variant,
                "payload_preview": build_payload_preview(
                    symbol.payload,
                    max_length=max_payload_preview_length,
                ),
                "confidence": symbol.confidence,
            }
            for symbol in symbols
        ]

    def build_stats(self) -> dict[str, Any]:
        duration_ms = 0
        if self.finished_at is not None:
            duration_ms = max(
                int(round((self.finished_at - self.started_at).total_seconds() * 1000)),
                0,
            )

        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "frames_processed": self.frames_processed,
            "sampled_frames_count": len(self.sampled_frames),
            "debug_frames_count": len(self.debug_frame_keys),
            "debug_overlays_count": len(self.debug_overlay_keys),
            "debug_crops_count": len(self.debug_crop_keys),
            "frame_count": self.video_metadata.frame_count or 0,
            "detections_total": len(self.detections),
            "detections_by_frame_count": len(self.detections_by_frame()),
            "crops_total": len(self.crop_candidates),
            "decode_attempts_total": len(self.decode_attempts),
            "decoded_symbols_total": len(self.decoded_symbols),
            "decoded_qr_total": self.decoded_symbols_by_type()["qr"],
            "decoded_barcode_total": self.decoded_symbols_by_type()["barcode"],
            "final_rows": len(self.csv_rows),
            "duration_ms": duration_ms,
        }


class PipelineExecutionError(RuntimeError):
    def __init__(self, pipeline_name: str, stage_name: str, message: str) -> None:
        super().__init__(
            f"Pipeline '{pipeline_name}' failed at stage '{stage_name}': {message}"
        )
        self.pipeline_name = pipeline_name
        self.stage_name = stage_name
        self.message = message


class BaseStage(ABC):
    name: str
    always_run: bool = False

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {}

    @abstractmethod
    def run(self, context: PipelineContext) -> StageOutcome:
        raise NotImplementedError


class BasePipeline(ABC):
    name: str
    default_version: str

    @abstractmethod
    def run(self, request: Any, storage: Any) -> Any:
        raise NotImplementedError
