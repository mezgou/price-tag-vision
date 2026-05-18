from __future__ import annotations

from typing import Any

from app.pipelines.base import PipelineContext, StageOutcome
from app.pipelines.price_tag_v3.stages.frame_sampling import V3FrameSamplingStage


class V4FrameSamplingStage(V3FrameSamplingStage):
    name = "V4FrameSamplingStage"

    def run(self, context: PipelineContext) -> StageOutcome:
        outcome = super().run(context)
        context.artifacts["v4_frame_sampling"] = {
            "sampled_frame_indices": [frame.frame_index for frame in context.sampled_frames],
            "sampled_timestamps_ms": [frame.timestamp_ms for frame in context.sampled_frames],
            "sampled_frames_count": len(context.sampled_frames),
        }
        output_summary: dict[str, Any] = dict(outcome.output_summary)
        output_summary["manifested_frame_indices"] = len(context.sampled_frames)
        return StageOutcome(
            output_summary=output_summary,
            warnings=outcome.warnings,
            errors=outcome.errors,
        )
