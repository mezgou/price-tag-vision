from __future__ import annotations

import json

from app.pipelines.base import StageExecution
from app.pipelines.price_tag_cpu_v1.stages.debug_manifest import DebugManifestStage
from app.utils.timing import utc_now


def test_debug_manifest_contains_expected_structure(
    pipeline_context,
    in_memory_storage,
) -> None:
    pipeline_context.finished_at = utc_now()
    pipeline_context.stage_reports.append(
        StageExecution(
            name="FrameMetadataStage",
            status="succeeded",
            duration_ms=5,
            input_summary={"input_video_key": pipeline_context.input_video_key},
            output_summary={"video_metadata": pipeline_context.video_metadata.to_dict()},
        )
    )

    stage = DebugManifestStage()
    outcome = stage.run(pipeline_context)

    manifest_key = outcome.output_summary["manifest_key"]
    payload, _ = in_memory_storage.objects[manifest_key]
    manifest = json.loads(payload.decode("utf-8"))

    assert manifest["job_id"] == pipeline_context.job_id
    assert manifest["pipeline_name"] == pipeline_context.pipeline_name
    assert manifest["pipeline_version"] == pipeline_context.pipeline_version
    assert isinstance(manifest["started_at"], str)
    assert isinstance(manifest["finished_at"], str)
    assert manifest["video_metadata"]["filename"] == "input.mp4"
    assert isinstance(manifest["config"], dict)
    assert isinstance(manifest["artifacts"], dict)
    assert isinstance(manifest["stats"], dict)
    assert isinstance(manifest["errors"], list)
    assert isinstance(manifest["warnings"], list)
    assert manifest["stats"]["debug_overlays_count"] == 0
    assert manifest["stats"]["debug_crops_count"] == 0
    assert manifest["stats"]["detections_by_frame_count"] == 0
    assert manifest["stats"]["crops_total"] == 0
    assert manifest["stats"]["decode_attempts_total"] == 0
    assert manifest["stats"]["decoded_symbols_total"] == 0
    assert manifest["stages"][0]["name"] == "FrameMetadataStage"
    assert manifest["stages"][-1]["name"] == "DebugManifestStage"
    assert isinstance(manifest["stages"][-1]["warnings"], list)
    assert isinstance(manifest["stages"][-1]["errors"], list)
