from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.utils.video_metadata import read_video_metadata


class FrameMetadataStage(BaseStage):
    name = "FrameMetadataStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "input_video_key": context.input_video_key,
            "local_video_path": str(context.local_video_path),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        video_metadata, warnings = read_video_metadata(
            context.local_video_path,
            input_video_key=context.input_video_key,
        )
        context.video_metadata = video_metadata
        return StageOutcome(
            output_summary={
                "video_metadata": video_metadata.to_dict(),
                "warnings_added": len(warnings),
            },
            warnings=warnings,
        )
