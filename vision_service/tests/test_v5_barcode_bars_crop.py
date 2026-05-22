from __future__ import annotations

import numpy as np

from app.pipelines.price_tag_v5.stages import barcode_bars_crop as stage_module
from app.pipelines.price_tag_v5.stages.barcode_bars_crop import (
    V5BarcodeBarsCropDecodeStage,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality


def test_v5_barcode_bars_crop_decode_adds_valid_ean13(
    pipeline_context,
    monkeypatch,
) -> None:
    crop = _crop()
    pipeline_context.crop_candidates = [crop]
    pipeline_context.config["v5_barcode_bars_decode"] = {
        "enabled": True,
        "max_crops": 10,
        "min_confidence": 0.74,
    }
    monkeypatch.setattr(
        stage_module,
        "_load_crop_image",
        lambda *, crop, frame_lookup: np.zeros((80, 160, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(stage_module, "_barcode_band", lambda image: image)
    monkeypatch.setattr(stage_module, "_barcode_variants", lambda image: [("fake", image)])
    monkeypatch.setattr(stage_module, "_decode_zxingcpp", lambda image: [])
    monkeypatch.setattr(stage_module, "_decode_pyzbar", lambda image: [])
    monkeypatch.setattr(stage_module, "_decode_signal", lambda image: [("8051070512049", 0.88)])

    outcome = V5BarcodeBarsCropDecodeStage().run(pipeline_context)

    assert outcome.output_summary["hits"] == 1
    assert pipeline_context.decoded_symbols[0].payload == "8051070512049"
    assert pipeline_context.decoded_symbols[0].symbol_type == "barcode"
    assert pipeline_context.decoded_symbols[0].attributes["source"] == "v5_crop_barcode_bars"


def test_v5_barcode_bars_crop_decode_rejects_invalid_ean13(
    pipeline_context,
    monkeypatch,
) -> None:
    pipeline_context.crop_candidates = [_crop()]
    monkeypatch.setattr(
        stage_module,
        "_load_crop_image",
        lambda *, crop, frame_lookup: np.zeros((80, 160, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(stage_module, "_barcode_band", lambda image: image)
    monkeypatch.setattr(stage_module, "_barcode_variants", lambda image: [("fake", image)])
    monkeypatch.setattr(stage_module, "_decode_zxingcpp", lambda image: [])
    monkeypatch.setattr(stage_module, "_decode_pyzbar", lambda image: [])
    monkeypatch.setattr(stage_module, "_decode_signal", lambda image: [("8051070512040", 0.99)])

    outcome = V5BarcodeBarsCropDecodeStage().run(pipeline_context)

    assert outcome.output_summary["hits"] == 0
    assert pipeline_context.decoded_symbols == []


def _crop() -> CropCandidate:
    bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60)
    return CropCandidate(
        crop_id="crop_1",
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        bbox=bbox,
        padded_bbox=bbox,
        crop_key="",
        width=100,
        height=60,
        quality=CropQuality(
            sharpness=1.0,
            brightness=100.0,
            contrast=20.0,
            glare_ratio=0.0,
            area_ratio=0.1,
            score=0.9,
        ),
        source="test",
        attributes={"track_id": "track_1"},
    )
