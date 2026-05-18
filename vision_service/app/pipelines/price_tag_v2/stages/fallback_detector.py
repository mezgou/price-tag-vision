from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.heuristic_candidate_detection import (
    CandidateDetectionConfig,
    HeuristicCandidateDetectionStage,
)


class FallbackHeuristicCandidateDetectionStage(BaseStage):
    name = "FallbackHeuristicCandidateDetectionStage"

    def __init__(self) -> None:
        self._fallback_stage = HeuristicCandidateDetectionStage()

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = CandidateDetectionConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "fallback_only": _fallback_only(context),
            "existing_detections_count": len(context.detections),
            "sampled_frames_count": len(context.sampled_frames),
            "source": config.source,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        if _fallback_only(context) and context.detections:
            return StageOutcome(
                output_summary={
                    "enabled": True,
                    "skipped": True,
                    "reason": "YOLO already produced detections.",
                    "existing_detections_count": len(context.detections),
                }
            )

        outcome = self._fallback_stage.run(context)
        return StageOutcome(
            output_summary={
                **outcome.output_summary,
                "fallback_only": _fallback_only(context),
            },
            warnings=outcome.warnings,
            errors=outcome.errors,
        )


def _fallback_only(context: PipelineContext) -> bool:
    raw_config = context.config.get("candidate_detection", {})
    if not isinstance(raw_config, dict):
        return True
    value = raw_config.get("fallback_only", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return True

