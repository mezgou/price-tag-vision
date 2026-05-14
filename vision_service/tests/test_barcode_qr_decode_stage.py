from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import (
    BarcodeQrDecodeStage,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality


def test_barcode_qr_decode_stage_handles_empty_crops(
    pipeline_context,
) -> None:
    pipeline_context.config["barcode_qr_decode"] = {"enabled": True}
    pipeline_context.crop_candidates = []

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decode_attempts_count"] == 0
    assert outcome.output_summary["decoded_symbols_count"] == 0
    assert pipeline_context.decode_attempts == []
    assert pipeline_context.decoded_symbols == []


def test_barcode_qr_decode_stage_handles_crop_without_symbols(
    pipeline_context,
    tmp_path: Path,
) -> None:
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (40, 70), (160, 130), (245, 245, 245), thickness=-1)
    frame_path = tmp_path / "runtime" / "frame_no_qr.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["barcode_qr_decode"] = {
        "enabled": True,
        "max_crops": 10,
        "min_crop_quality_score": 0.0,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=0,
            timestamp_ms=0,
            original_width=200,
            original_height=200,
            processed_width=200,
            processed_height=200,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="crop_1",
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=0,
            bbox=BoundingBox(x_min=40, y_min=70, x_max=160, y_max=130),
            padded_bbox=BoundingBox(x_min=40, y_min=70, x_max=160, y_max=130),
            crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
            width=120,
            height=60,
            quality=CropQuality(
                sharpness=10.0,
                brightness=220.0,
                contrast=12.0,
                glare_ratio=0.0,
                area_ratio=0.18,
                score=0.6,
            ),
            source="crop_extraction_v1",
        )
    ]

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decode_attempts_count"] > 0
    assert outcome.output_summary["decoded_symbols_count"] == 0
    assert len(pipeline_context.decode_attempts) > 0
    assert pipeline_context.decoded_symbols == []


def test_barcode_qr_decode_stage_decodes_synthetic_qr_image(
    pipeline_context,
    tmp_path: Path,
) -> None:
    qr_payload = "https://example.test/price-tag-vision"
    qr_image = _build_qr_image(qr_payload, target_size=120)

    frame = np.full((220, 220, 3), 255, dtype=np.uint8)
    frame[50:170, 50:170] = qr_image
    frame_path = tmp_path / "runtime" / "frame_qr.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["barcode_qr_decode"] = {
        "enabled": True,
        "max_crops": 10,
        "min_crop_quality_score": 0.0,
        "stop_after_first_success_per_crop": False,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=5,
            timestamp_ms=1000,
            original_width=220,
            original_height=220,
            processed_width=220,
            processed_height=220,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="crop_qr",
            detection_id="det_qr",
            frame_index=5,
            timestamp_ms=1000,
            bbox=BoundingBox(x_min=50, y_min=50, x_max=170, y_max=170),
            padded_bbox=BoundingBox(x_min=45, y_min=45, x_max=175, y_max=175),
            crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
            width=130,
            height=130,
            quality=CropQuality(
                sharpness=1000.0,
                brightness=180.0,
                contrast=90.0,
                glare_ratio=0.0,
                area_ratio=0.35,
                score=0.95,
            ),
            source="crop_extraction_v1",
        )
    ]

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decoded_symbols_count"] >= 1
    assert outcome.output_summary["decoded_symbols_by_type"]["qr"] >= 1
    assert len(pipeline_context.decode_attempts) > 0
    assert len(pipeline_context.decoded_symbols) >= 1
    assert any(symbol.payload == qr_payload for symbol in pipeline_context.decoded_symbols)
    assert all(symbol.symbol_type == "qr" for symbol in pipeline_context.decoded_symbols)
    assert len({symbol.payload for symbol in pipeline_context.decoded_symbols}) == 1


def _build_qr_image(payload: str, *, target_size: int) -> np.ndarray:
    params = cv2.QRCodeEncoder_Params()
    encoder = cv2.QRCodeEncoder_create(params)
    qr = encoder.encode(payload)
    qr = cv2.resize(qr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
