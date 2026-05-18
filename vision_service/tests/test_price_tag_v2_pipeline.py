from __future__ import annotations

from app.pipelines.price_tag_v2.pipeline import PriceTagV2Pipeline
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload


def test_price_tag_v2_pipeline_config_loads() -> None:
    pipeline = PriceTagV2Pipeline()

    assert pipeline.name == "price_tag_v2"
    assert pipeline.default_version == "0.3.0"
    assert pipeline._base_config["frame_sampling"]["sample_fps"] == 5.0
    assert pipeline._base_config["frame_sampling"]["orientation_mode"] == "none"
    assert pipeline._base_config["yolo_detection"]["enabled"] is True
    assert pipeline._base_config["yolo_detection"]["device"] == "auto"
    assert pipeline._base_config["candidate_detection"]["fallback_only"] is True
    assert pipeline._base_config["row_fusion"]["enabled"] is True


def test_parse_qr_payload_maps_compact_fields() -> None:
    parsed = parse_qr_payload("barcode=4607124143901;p1=73.99;p4=69,99;wL1C=2;wL1P=65.00")

    assert parsed == {
        "qr_code_barcode": "4607124143901",
        "price1_qr": "73.99",
        "price4_qr": "69.99",
        "wholesale_level_1_count": "2",
        "wholesale_level_1_price": "65.00",
    }
