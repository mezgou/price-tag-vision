from __future__ import annotations

import numpy as np
import pytest

from app.pipelines.price_tag_v2.pipeline import PriceTagV2Pipeline
from app.pipelines.price_tag_v2.stages.row_fusion import (
    _FusionGroup,
    _fuse_group_to_row,
    parse_qr_payload,
)
from app.pipelines.price_tag_v2.stages.zonal_ocr import classify_tag_color
from app.schemas.detections import BoundingBox, DetectionCandidate

NO_VALUE = "\u043d\u0435\u0442"


def test_price_tag_v2_pipeline_config_loads() -> None:
    pipeline = PriceTagV2Pipeline()

    assert pipeline.name == "price_tag_v2"
    assert pipeline.default_version == "0.3.0"
    assert pipeline._base_config["frame_sampling"]["sample_fps"] == 5.0
    assert pipeline._base_config["frame_sampling"]["orientation_mode"] == "none"
    assert pipeline._base_config["yolo_detection"]["enabled"] is True
    assert pipeline._base_config["yolo_detection"]["device"] == "0"
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


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (np.full((60, 120, 3), (0, 0, 255), dtype=np.uint8), "\u043a\u0440\u0430\u0441\u043d\u044b\u0439"),
        (np.full((60, 120, 3), (0, 255, 255), dtype=np.uint8), "\u0436\u0451\u043b\u0442\u044b\u0439"),
        (np.full((60, 120, 3), (255, 255, 255), dtype=np.uint8), "\u0431\u0435\u043b\u044b\u0439"),
        (np.zeros((60, 120, 3), dtype=np.uint8), ""),
    ],
)
def test_classify_tag_color_maps_to_russian_contract(
    image: np.ndarray,
    expected: str,
) -> None:
    assert classify_tag_color(image) == expected


def test_classify_tag_color_keeps_red_over_large_white_area() -> None:
    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    image[30:, :, :] = (0, 0, 255)

    assert classify_tag_color(image) == "\u043a\u0440\u0430\u0441\u043d\u044b\u0439"


def test_row_fusion_defaults_missing_color_to_no_value(pipeline_context) -> None:
    detection = DetectionCandidate(
        detection_id="det-1",
        frame_index=0,
        timestamp_ms=1234,
        label="price_tag",
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=90),
        confidence=0.95,
        source="unit-test",
        attributes={},
    )
    group = _FusionGroup(key="det-1", detections=[detection])

    row = _fuse_group_to_row(
        context=pipeline_context,
        group=group,
        default_absent_value=NO_VALUE,
    )

    assert row["color"] == NO_VALUE
