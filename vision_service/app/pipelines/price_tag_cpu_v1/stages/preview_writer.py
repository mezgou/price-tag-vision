from __future__ import annotations

from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import BarcodeQrDecodeConfig
from app.pipelines.price_tag_cpu_v1.stages.crop_extraction import CropExtractionConfig
from app.pipelines.price_tag_cpu_v1.stages.frame_sampling import FrameSamplingConfig
from app.pipelines.price_tag_cpu_v1.stages.heuristic_candidate_detection import (
    CandidateDetectionConfig,
)


class PreviewWriterStage(BaseStage):
    name = "PreviewWriterStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "detections_count": len(context.detections),
            "detections_by_frame_count": len(context.detections_by_frame()),
            "rows_count": len(context.csv_rows),
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
            "crops_count": len(context.crop_candidates),
            "decode_attempts_count": len(context.decode_attempts),
            "decoded_symbols_count": len(context.decoded_symbols),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        frame_sampling_config = FrameSamplingConfig.from_context(context)
        detection_config = CandidateDetectionConfig.from_context(context)
        crop_config = CropExtractionConfig.from_context(context)
        decode_config = BarcodeQrDecodeConfig.from_context(context)
        stats = context.build_stats()
        sampled_frames_preview = [
            frame.to_dict() for frame in context.sampled_frames[:5]
        ]
        context.artifacts["debug_frame_keys"] = list(context.debug_frame_keys)
        context.artifacts["debug_overlay_keys"] = list(context.debug_overlay_keys)
        context.artifacts["debug_crop_keys"] = list(context.debug_crop_keys)
        context.artifacts["debug_mask_keys"] = list(context.debug_mask_keys)
        context.artifacts["debug_contact_sheet_keys"] = list(context.debug_contact_sheet_keys)
        preview_payload = {
            "job_id": context.job_id,
            "pipeline_name": context.pipeline_name,
            "pipeline_version": context.pipeline_version,
            "video_metadata": context.video_metadata.to_dict(),
            "frames_processed": context.frames_processed,
            "sampled_frames_count": len(context.sampled_frames),
            "debug_frame_keys": list(context.debug_frame_keys),
            "debug_overlay_keys": list(context.debug_overlay_keys),
            "debug_mask_keys": list(context.debug_mask_keys),
            "debug_masks_count": stats["debug_masks_count"],
            "sampled_frames": sampled_frames_preview,
            "detections_count": len(context.detections),
            "detections_by_frame": context.detections_by_frame(),
            "sample_detections": context.serialize_detections(limit=10),
            "crops_count": len(context.crop_candidates),
            "debug_crop_keys": list(context.debug_crop_keys),
            "debug_contact_sheet_keys": list(context.debug_contact_sheet_keys),
            "debug_contact_sheets_count": stats["debug_contact_sheets_count"],
            "crop_quality_summary": stats["crop_quality_summary"],
            "sample_crops": context.serialize_crops(
                limit=crop_config.top_crops_preview_limit,
                sort_by_quality=True,
            ),
            "decode_attempts_count": len(context.decode_attempts),
            "decoded_symbols_count": len(context.decoded_symbols),
            "decoded_symbols_by_type": context.decoded_symbols_by_type(),
            "sample_decoded_symbols": context.serialize_decoded_symbols_preview(
                limit=10,
                max_payload_preview_length=decode_config.max_payload_preview_length,
            ),
            "ocr_summary": context.artifacts.get("ocr_summary", {}),
            "rows_count": len(context.csv_rows),
            "message": (
                "Price tag pipeline completed frame sampling, detection, crop extraction, "
                "QR/barcode decode, OCR, and row fusion stages."
            ),
            "artifacts": {
                "csv_key": context.artifacts.get("csv_key", context.artifact_writer.csv_key),
                "preview_key": context.artifact_writer.preview_key,
                "manifest_key": context.artifact_writer.manifest_key,
                "crop_keys": list(context.artifacts.get("crop_keys", [])),
                "debug_frame_keys": list(context.debug_frame_keys),
                "debug_overlay_keys": list(context.debug_overlay_keys),
                "debug_crop_keys": list(context.debug_crop_keys),
                "debug_mask_keys": list(context.debug_mask_keys),
                "debug_contact_sheet_keys": list(context.debug_contact_sheet_keys),
            },
            "debug_status": {
                "frames_enabled": frame_sampling_config.debug_save_frames,
                "overlays_enabled": detection_config.debug_save_overlays,
                "masks_enabled": detection_config.debug_save_masks,
                "rejected_preview_enabled": detection_config.debug_save_rejected,
                "crops_enabled": crop_config.debug_save_crops,
                "contact_sheet_enabled": crop_config.debug_save_contact_sheet,
            },
            "summary": stats,
            "rows": context.csv_rows[:25],
        }

        preview_key = context.artifact_writer.upload_json("preview.json", preview_payload)
        context.artifacts["preview_key"] = preview_key

        return StageOutcome(
            output_summary={
                "preview_key": preview_key,
                "rows_count": len(context.csv_rows),
                "sampled_frames_count": len(context.sampled_frames),
                "detections_count": len(context.detections),
                "crops_count": len(context.crop_candidates),
                "decode_attempts_count": len(context.decode_attempts),
                "decoded_symbols_count": len(context.decoded_symbols),
            }
        )
