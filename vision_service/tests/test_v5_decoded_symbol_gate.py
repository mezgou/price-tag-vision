from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_v5.stages.decoded_symbol_gate import (
    V5DecodedSymbolGateStage,
    _normalized_catalog_codes,
)
from app.schemas.detections import BoundingBox, DecodedSymbol, DetectionCandidate


def test_decoded_symbol_gate_keeps_valid_catalog_barcode(
    pipeline_context,
    tmp_path: Path,
) -> None:
    db_path = _db(tmp_path)
    pipeline_context.config["v5_decoded_symbol_gate"] = {"db_hack_path": str(db_path)}
    pipeline_context.detections = [_detection()]
    pipeline_context.decoded_symbols = [_symbol("8051070512049", symbol_type="barcode")]

    outcome = V5DecodedSymbolGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_symbols"] == 1
    assert pipeline_context.decoded_symbols[0].payload == "8051070512049"
    evidence = pipeline_context.artifacts["v5_code_evidence_by_track"]["track_1"][0]
    assert evidence["code"] == "8051070512049"
    assert evidence["catalog_product_name"] == "Wine A"


def test_decoded_symbol_gate_rejects_bad_checksum(
    pipeline_context,
    tmp_path: Path,
) -> None:
    db_path = _db(tmp_path)
    pipeline_context.config["v5_decoded_symbol_gate"] = {"db_hack_path": str(db_path)}
    pipeline_context.detections = [_detection()]
    pipeline_context.decoded_symbols = [_symbol("8051070512040", symbol_type="barcode")]

    outcome = V5DecodedSymbolGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_symbols"] == 0
    assert outcome.output_summary["rejected_symbols"] == 1
    assert pipeline_context.decoded_symbols == []


def test_decoded_symbol_gate_strips_unsafe_qr_barcode_but_keeps_prices(
    pipeline_context,
    tmp_path: Path,
) -> None:
    db_path = _db(tmp_path)
    pipeline_context.config["v5_decoded_symbol_gate"] = {"db_hack_path": str(db_path)}
    pipeline_context.detections = [_detection()]
    pipeline_context.decoded_symbols = [
        _symbol("barcode=8051070512040;p1=1299.99;p4=999.99", symbol_type="qr")
    ]

    outcome = V5DecodedSymbolGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_symbols"] == 1
    assert "barcode" not in pipeline_context.decoded_symbols[0].payload
    assert "p1=1299.99" in pipeline_context.decoded_symbols[0].payload
    assert "p4=999.99" in pipeline_context.decoded_symbols[0].payload


def test_catalog_codes_include_valid_ean13_aliases() -> None:
    codes = _normalized_catalog_codes({"036000291452": "Wine A"})

    assert codes["0036000291452"] == "Wine A"


def _db(tmp_path: Path) -> Path:
    db_path = tmp_path / "db_hack.csv"
    db_path.write_bytes(
        "fullname;code\r\nWine A;8051070512049\r\n".encode("cp1251")
    )
    return db_path


def _detection() -> DetectionCandidate:
    return DetectionCandidate(
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        label="price_tag_candidate",
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60),
        confidence=0.9,
        source="test",
        attributes={"track_id": "track_1"},
    )


def _symbol(payload: str, *, symbol_type: str) -> DecodedSymbol:
    return DecodedSymbol(
        symbol_id="symbol_1",
        crop_id="crop_1",
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        symbol_type=symbol_type,
        decoder="test",
        variant="test",
        payload=payload,
        confidence=0.9,
    )
