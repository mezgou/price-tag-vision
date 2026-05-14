from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome


class EmptyDetectionsStage(BaseStage):
    name = "EmptyDetectionsStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "video_metadata_ready": context.video_metadata.frame_count is not None
            or context.video_metadata.width is not None,
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        context.detections = []
        context.csv_rows = []
        return StageOutcome(
            output_summary={
                "detections_count": len(context.detections),
                "rows_buffered": len(context.csv_rows),
            }
        )
