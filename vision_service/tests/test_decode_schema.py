from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.detections import BoundingBox, DecodeAttempt, DecodedSymbol


def test_decode_attempt_schema_accepts_valid_payload() -> None:
    attempt = DecodeAttempt(
        attempt_id="crop_1__opencv_qr_detector__original",
        crop_id="crop_1",
        decoder="opencv_qr_detector",
        variant="original",
        success=True,
        duration_ms=12,
    )

    assert attempt.success is True
    assert attempt.duration_ms == 12


def test_decoded_symbol_schema_accepts_valid_payload() -> None:
    symbol = DecodedSymbol(
        symbol_id="crop_1__symbol_01",
        crop_id="crop_1",
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        symbol_type="qr",
        decoder="opencv_qr_detector",
        variant="resized_x2",
        payload="https://example.test",
        confidence=0.78,
        bbox=BoundingBox(x_min=4, y_min=5, x_max=80, y_max=81),
    )

    assert symbol.symbol_type == "qr"
    assert symbol.confidence == 0.78


def test_decoded_symbol_schema_rejects_invalid_symbol_type() -> None:
    with pytest.raises(ValidationError):
        DecodedSymbol(
            symbol_id="crop_1__symbol_01",
            crop_id="crop_1",
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=0,
            symbol_type="datamatrix",
            decoder="opencv_qr_detector",
            variant="original",
            payload="payload",
            confidence=0.5,
        )
