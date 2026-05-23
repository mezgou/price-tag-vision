from __future__ import annotations

from app.pipelines.price_tag_v5.pipeline import PriceTagV5Pipeline


def test_price_tag_v5_pipeline_config_uses_balanced_final_gate() -> None:
    pipeline = PriceTagV5Pipeline()

    assert pipeline.name == "price_tag_v5"
    assert pipeline.default_version == "0.1.0"
    assert pipeline._base_config["camera"]["undistort"] is True
    gate_config = pipeline._base_config["v5_row_confidence_gate"]
    assert gate_config["mode"] == "balanced"
    assert gate_config["min_confidence"] == 0.40
    assert gate_config["suppress_weak_when_identity_present"] is True
    assert gate_config["identity_present_min_confidence"] == 0.45


def test_price_tag_v5_gate_runs_before_csv_writer() -> None:
    pipeline = PriceTagV5Pipeline()

    stage_names = [stage.name for stage in pipeline._stages]
    assert "BarcodeQrDecodeStage" in stage_names
    assert "QrZoneDecodeStage" in stage_names
    assert "V5BarcodeBarsCropDecodeStage" in stage_names
    assert "V5DecodedSymbolGateStage" in stage_names
    assert stage_names.index("BarcodeQrDecodeStage") < stage_names.index("V5BlockOcrCatalogStage")
    assert stage_names.index("V5DecodedSymbolGateStage") < stage_names.index("V5BlockOcrCatalogStage")
    assert stage_names.index("V5RowConfidenceGateStage") < stage_names.index("CsvWriterStage")


def test_price_tag_v5_config_enables_barcode_qr_sku_path() -> None:
    pipeline = PriceTagV5Pipeline()

    assert pipeline._base_config["barcode_qr_decode"]["enabled"] is True
    assert pipeline._base_config["barcode_qr_zone_decode"]["enabled"] is True
    assert (
        pipeline._base_config["barcode_qr_zone_decode"]["skip_if_code_evidence_present"]
        is True
    )
    assert pipeline._base_config["v5_barcode_bars_decode"]["enabled"] is True
    assert (
        pipeline._base_config["v5_barcode_bars_decode"]["skip_if_code_evidence_present"]
        is True
    )
    assert pipeline._base_config["v5_decoded_symbol_gate"]["enabled"] is True
