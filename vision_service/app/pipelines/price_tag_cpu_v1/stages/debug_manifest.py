from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageExecution, StageOutcome


class DebugManifestStage(BaseStage):
    name = "DebugManifestStage"
    always_run = True

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "stages_recorded": len(context.stage_reports),
            "errors_count": len(context.errors),
            "warnings_count": len(context.warnings),
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
            "detections_count": len(context.detections),
            "crops_count": len(context.crop_candidates),
            "decode_attempts_count": len(context.decode_attempts),
            "decoded_symbols_count": len(context.decoded_symbols),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        manifest_key = context.artifact_writer.manifest_key
        synthetic_stage = StageExecution(
            name=self.name,
            status="succeeded",
            duration_ms=0,
            input_summary=self.describe_input(context),
            output_summary={
                "manifest_key": manifest_key,
                "sampled_frames_count": len(context.sampled_frames),
                "debug_frames_count": len(context.debug_frame_keys),
                "debug_overlays_count": len(context.debug_overlay_keys),
                "debug_masks_count": len(context.debug_mask_keys),
                "detections_count": len(context.detections),
                "debug_crops_count": len(context.debug_crop_keys),
                "debug_contact_sheets_count": len(context.debug_contact_sheet_keys),
                "crops_count": len(context.crop_candidates),
                "crop_quality_summary": context.build_stats()["crop_quality_summary"],
                "decode_attempts_count": len(context.decode_attempts),
                "decoded_symbols_count": len(context.decoded_symbols),
            },
        )
        payload = context.build_manifest(
            extra_stages=[synthetic_stage],
            finished_at=context.finished_at,
        )
        context.artifact_writer.upload_json("debug/pipeline_manifest.json", payload)
        context.artifacts["manifest_key"] = manifest_key

        return StageOutcome(
            output_summary={
                "manifest_key": manifest_key,
                "stages_count": len(payload["stages"]),
                "sampled_frames_count": len(context.sampled_frames),
            }
        )
