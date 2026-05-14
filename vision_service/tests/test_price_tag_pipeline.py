from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from app.pipelines.price_tag_cpu_v1.pipeline import PriceTagCpuV1Pipeline
from tests.video_factory import write_synthetic_video


def test_price_tag_pipeline_config_loads() -> None:
    pipeline = PriceTagCpuV1Pipeline()

    assert pipeline.name == "price_tag_cpu_v1"
    assert pipeline.default_version == "0.1.0"
    assert pipeline._base_config["frame_sampling"]["sample_fps"] == 1.0
    assert pipeline._base_config["frame_sampling"]["orientation_mode"] == "rotate_90_ccw"
    assert pipeline._base_config["candidate_detection"]["enabled"] is True
    assert pipeline._base_config["crop_extraction"]["enabled"] is True
    assert pipeline._base_config["barcode_qr_decode"]["enabled"] is True


def test_process_api_returns_sampling_stats_and_artifacts(
    app_client,
    in_memory_storage,
    tmp_path: Path,
) -> None:
    video_path = write_synthetic_video(tmp_path / "inputs" / "api-sample.mp4")
    input_video_key = "inputs/job-api-123/api-sample.mp4"
    in_memory_storage.downloads[input_video_key] = video_path.read_bytes()

    response = app_client.post(
        "/api/pipeline/process",
        json={
            "job_id": "job-api-123",
            "input_video_key": input_video_key,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-api-123"
    assert body["status"] == "succeeded"
    assert body["csv_key"] == "outputs/job-api-123/result.csv"
    assert body["preview_key"] == "outputs/job-api-123/preview.json"
    assert body["crop_keys"] == []
    assert body["stats"]["pipeline_name"] == "price_tag_cpu_v1"
    assert body["stats"]["pipeline_version"] == "0.1.0"
    assert body["stats"]["frames_processed"] > 0
    assert body["stats"]["sampled_frames_count"] > 0
    assert body["stats"]["debug_frames_count"] > 0
    assert body["stats"]["debug_overlays_count"] > 0
    assert body["stats"]["debug_crops_count"] >= 0
    assert body["stats"]["frame_count"] == 10
    assert body["stats"]["detections_total"] >= 0
    assert body["stats"]["detections_by_frame_count"] >= 0
    assert body["stats"]["crops_total"] >= 0
    assert body["stats"]["decode_attempts_total"] >= 0
    assert body["stats"]["decoded_symbols_total"] >= 0
    assert body["stats"]["decoded_qr_total"] >= 0
    assert body["stats"]["decoded_barcode_total"] >= 0
    assert body["stats"]["final_rows"] == 0

    csv_payload, csv_content_type = in_memory_storage.objects["outputs/job-api-123/result.csv"]
    reader = csv.reader(io.StringIO(csv_payload.decode("utf-8")))
    rows = list(reader)
    assert csv_content_type == "text/csv; charset=utf-8"
    assert len(rows) == 1

    preview_payload, _ = in_memory_storage.objects["outputs/job-api-123/preview.json"]
    preview = json.loads(preview_payload.decode("utf-8"))
    assert preview["pipeline_name"] == "price_tag_cpu_v1"
    assert preview["frames_processed"] > 0
    assert preview["sampled_frames_count"] > 0
    assert len(preview["debug_frame_keys"]) > 0
    assert len(preview["debug_overlay_keys"]) > 0
    assert isinstance(preview["debug_crop_keys"], list)
    assert preview["rows_count"] == 0
    assert preview["detections_count"] >= 0
    assert preview["crops_count"] >= 0
    assert preview["decode_attempts_count"] >= 0
    assert preview["decoded_symbols_count"] >= 0
    assert preview["video_metadata"]["frame_count"] == 10
    assert isinstance(preview["detections_by_frame"], dict)
    assert isinstance(preview["sample_detections"], list)
    assert isinstance(preview["sample_crops"], list)
    assert isinstance(preview["decoded_symbols_by_type"], dict)
    assert isinstance(preview["sample_decoded_symbols"], list)

    manifest_payload, _ = in_memory_storage.objects[
        "outputs/job-api-123/debug/pipeline_manifest.json"
    ]
    manifest = json.loads(manifest_payload.decode("utf-8"))
    stage_names = [stage["name"] for stage in manifest["stages"]]
    assert manifest["pipeline_name"] == "price_tag_cpu_v1"
    assert "FrameSamplingStage" in stage_names
    assert "HeuristicCandidateDetectionStage" in stage_names
    assert "CropExtractionStage" in stage_names
    assert "BarcodeQrDecodeStage" in stage_names

    frame_sampling_stage = next(
        stage for stage in manifest["stages"] if stage["name"] == "FrameSamplingStage"
    )
    assert frame_sampling_stage["status"] == "succeeded"
    assert frame_sampling_stage["output_summary"]["sampled_frames_count"] > 0
    assert frame_sampling_stage["output_summary"]["debug_frames_saved"] > 0
    assert frame_sampling_stage["output_summary"]["orientation_mode"] == "rotate_90_ccw"
    heuristic_stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "HeuristicCandidateDetectionStage"
    )
    assert heuristic_stage["status"] == "succeeded"
    assert heuristic_stage["output_summary"]["frames_processed"] > 0
    crop_stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "CropExtractionStage"
    )
    assert crop_stage["status"] == "succeeded"
    assert crop_stage["output_summary"]["crops_count"] >= 0
    decode_stage = next(
        stage
        for stage in manifest["stages"]
        if stage["name"] == "BarcodeQrDecodeStage"
    )
    assert decode_stage["status"] == "succeeded"
    assert decode_stage["output_summary"]["decode_attempts_count"] >= 0

    debug_frame_keys = sorted(
        key for key in in_memory_storage.objects if key.startswith("outputs/job-api-123/debug/frames/")
    )
    assert debug_frame_keys
    debug_overlay_keys = sorted(
        key
        for key in in_memory_storage.objects
        if key.startswith("outputs/job-api-123/debug/overlays/")
    )
    assert debug_overlay_keys
