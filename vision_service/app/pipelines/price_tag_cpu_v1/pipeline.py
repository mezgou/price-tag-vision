from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import yaml

from app.pipelines.base import (
    BasePipeline,
    BaseStage,
    PipelineContext,
    PipelineExecutionError,
    StageExecution,
    VideoMetadata,
)
from app.pipelines.price_tag_cpu_v1.stages import (
    BarcodeQrDecodeStage,
    CropExtractionStage,
    CsvWriterStage,
    DebugManifestStage,
    FrameMetadataStage,
    FrameSamplingStage,
    HeuristicCandidateDetectionStage,
    PreviewWriterStage,
)
from app.schemas.pipeline import ProcessRequest, ProcessResponse
from app.services.storage import ArtifactStorage
from app.utils.artifacts import ArtifactWriter
from app.utils.timing import elapsed_ms, utc_now


class PriceTagCpuV1Pipeline(BasePipeline):
    def __init__(self) -> None:
        self._config_path = Path(__file__).with_name("config.yaml")
        self._base_config = _load_yaml_config(self._config_path)
        pipeline_config = self._base_config.get("pipeline", {})
        self.name = str(pipeline_config.get("name", "price_tag_cpu_v1"))
        self.default_version = str(pipeline_config.get("version", "0.1.0"))
        self._stages: list[BaseStage] = [
            FrameMetadataStage(),
            FrameSamplingStage(),
            HeuristicCandidateDetectionStage(),
            CropExtractionStage(),
            BarcodeQrDecodeStage(),
            CsvWriterStage(),
            PreviewWriterStage(),
            DebugManifestStage(),
        ]

    def run(
        self,
        request: ProcessRequest,
        storage: ArtifactStorage,
    ) -> ProcessResponse:
        resolved_config = self._resolve_config(request)
        local_filename = Path(request.input_video_key).name or "input.mp4"

        with TemporaryDirectory(prefix=f"vision_{request.job_id}_") as temp_dir:
            work_dir = Path(temp_dir)
            local_video_path = work_dir / "inputs" / local_filename
            output_dir = work_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            storage.download_file(request.input_video_key, local_video_path)

            context = PipelineContext(
                job_id=request.job_id,
                input_video_key=request.input_video_key,
                pipeline_name=request.pipeline_name or self.name,
                pipeline_version=request.pipeline_version or self.default_version,
                config=resolved_config,
                local_video_path=local_video_path,
                work_dir=work_dir,
                output_dir=output_dir,
                artifact_writer=ArtifactWriter(storage, request.job_id),
                video_metadata=VideoMetadata.from_input_key(request.input_video_key),
                artifacts={
                    "csv_key": f"outputs/{request.job_id}/result.csv",
                    "preview_key": f"outputs/{request.job_id}/preview.json",
                    "manifest_key": f"outputs/{request.job_id}/debug/pipeline_manifest.json",
                    "crop_keys": [],
                    "debug_frame_keys": [],
                    "debug_overlay_keys": [],
                    "debug_crop_keys": [],
                    "debug_mask_keys": [],
                    "debug_contact_sheet_keys": [],
                },
            )

            failure: PipelineExecutionError | None = None

            for index, stage in enumerate(self._stages):
                if failure is not None and not stage.always_run:
                    context.stage_reports.append(
                        StageExecution(
                            name=stage.name,
                            status="skipped",
                            duration_ms=0,
                            input_summary={
                                "stage_index": index,
                                "reason": "Skipped because a previous stage failed.",
                            },
                            output_summary={},
                        )
                    )
                    continue

                if stage.always_run and context.finished_at is None:
                    context.finished_at = utc_now()

                input_summary = stage.describe_input(context)
                started = perf_counter()

                try:
                    outcome = stage.run(context)
                except Exception as exc:
                    error_payload = {
                        "stage": stage.name,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    context.errors.append(error_payload)
                    context.stage_reports.append(
                        StageExecution(
                            name=stage.name,
                            status="failed",
                            duration_ms=elapsed_ms(started),
                            input_summary=input_summary,
                            output_summary={},
                            errors=[error_payload],
                        )
                    )
                    if failure is None:
                        failure = PipelineExecutionError(
                            context.pipeline_name,
                            stage.name,
                            str(exc),
                        )
                    continue

                if outcome.warnings:
                    context.warnings.extend(outcome.warnings)
                if outcome.errors:
                    context.errors.extend(outcome.errors)

                context.stage_reports.append(
                    StageExecution(
                        name=stage.name,
                        status="succeeded",
                        duration_ms=elapsed_ms(started),
                        input_summary=input_summary,
                        output_summary=outcome.output_summary,
                        warnings=outcome.warnings,
                        errors=outcome.errors,
                    )
                )

            context.finished_at = context.finished_at or utc_now()

            try:
                self._write_final_manifest(context)
            except Exception as exc:
                context.errors.append(
                    {
                        "stage": "DebugManifestStage",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                if failure is None:
                    failure = PipelineExecutionError(
                        context.pipeline_name,
                        "DebugManifestStage",
                        str(exc),
                    )

            if failure is not None:
                raise failure

            return ProcessResponse(
                job_id=request.job_id,
                status="succeeded",
                csv_key=context.artifacts["csv_key"],
                preview_key=context.artifacts["preview_key"],
                crop_keys=list(context.artifacts.get("crop_keys", [])),
                stats=context.build_stats(),
            )

    def _resolve_config(self, request: ProcessRequest) -> dict[str, Any]:
        resolved = _deep_merge(deepcopy(self._base_config), request.config)
        pipeline_config = resolved.setdefault("pipeline", {})
        pipeline_config["name"] = request.pipeline_name or self.name
        pipeline_config["version"] = request.pipeline_version or self.default_version
        return resolved

    def _write_final_manifest(self, context: PipelineContext) -> None:
        manifest_payload = context.build_manifest(finished_at=context.finished_at)
        context.artifact_writer.upload_json("debug/pipeline_manifest.json", manifest_payload)
        context.artifacts["manifest_key"] = context.artifact_writer.manifest_key


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pipeline config must be a mapping: {config_path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            base[key] = _deep_merge(current, value)
        else:
            base[key] = value
    return base
