from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome


class PreviewWriterStage(BaseStage):
    name = "PreviewWriterStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "detections_count": len(context.detections),
            "rows_count": len(context.csv_rows),
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        sampled_frames_preview = [
            frame.to_dict() for frame in context.sampled_frames[:5]
        ]
        preview_payload = {
            "job_id": context.job_id,
            "pipeline_name": context.pipeline_name,
            "pipeline_version": context.pipeline_version,
            "video_metadata": context.video_metadata.to_dict(),
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
            "debug_frame_keys": list(context.debug_frame_keys),
            "sampled_frames": sampled_frames_preview,
            "detections_count": len(context.detections),
            "rows_count": len(context.csv_rows),
            "message": "Price tag pipeline completed frame sampling without detector or OCR stages.",
            "artifacts": {
                "csv_key": context.artifacts.get("csv_key", context.artifact_writer.csv_key),
                "preview_key": context.artifact_writer.preview_key,
                "manifest_key": context.artifact_writer.manifest_key,
                "crop_keys": list(context.artifacts.get("crop_keys", [])),
                "debug_frame_keys": list(context.debug_frame_keys),
            },
            "summary": {
                "frame_count": context.video_metadata.frame_count or 0,
                "frames_processed": context.frames_processed,
                "sampled_frames_count": len(context.sampled_frames),
                "debug_frames_count": len(context.debug_frame_keys),
                "detections_total": len(context.detections),
                "final_rows": len(context.csv_rows),
            },
            "rows": [],
        }

        preview_key = context.artifact_writer.upload_json("preview.json", preview_payload)
        context.artifacts["preview_key"] = preview_key

        return StageOutcome(
            output_summary={
                "preview_key": preview_key,
                "rows_count": len(context.csv_rows),
                "sampled_frames_count": len(context.sampled_frames),
            }
        )
