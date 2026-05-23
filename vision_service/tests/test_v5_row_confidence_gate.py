from __future__ import annotations

from app.pipelines.price_tag_v5.stages.row_confidence_gate import (
    RowConfidenceGateConfig,
    V5RowConfidenceGateStage,
    _row_confidence,
    _spatial_dedup,
    row_has_v5_evidence,
)


def _bbox_row(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    **fields: str,
) -> dict[str, str]:
    row = {
        "x_min": str(x_min),
        "y_min": str(y_min),
        "x_max": str(x_max),
        "y_max": str(y_max),
    }
    row.update(fields)
    return row


def test_row_gate_keeps_catalog_identity_match_key_and_price_rows() -> None:
    assert row_has_v5_evidence({"product_name": "Wine"})
    assert row_has_v5_evidence({"barcode": "8051070512049"})
    assert not row_has_v5_evidence({"barcode": "8051070512040"})
    assert row_has_v5_evidence({"price_card": "1199.99"})
    assert not row_has_v5_evidence({"catalog_guess_name": "Likely Wine"})


def test_row_gate_suppresses_absent_only_rows() -> None:
    row = {
        "product_name": "",
        "barcode": "",
        "qr_code_barcode": "",
        "price_card": "",
        "price_default": "",
        "code": "нет",
        "special_symbols": "нет",
    }

    assert not row_has_v5_evidence(row)


def test_strict_mode_filters_evidence_rows(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "strict",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        {"barcode": "", "product_name": "", "price_card": ""},
        {"barcode": "", "product_name": "", "price_card": "1499.99"},
        {"barcode": "8051070512049", "product_name": "", "price_card": ""},
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert len(pipeline_context.csv_rows) == 2
    assert outcome.output_summary["input_rows"] == 3
    assert outcome.output_summary["kept_rows"] == 2
    assert outcome.output_summary["suppressed_rows"] == 1


def test_recall_mode_keeps_every_localized_tag(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "recall",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(10, 10, 100, 100, color="red"),  # no identity, but localized
        _bbox_row(500, 500, 600, 600, product_name="Wine X"),
        {"product_name": "no bbox -> dropped"},
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_rows"] == 2
    assert outcome.output_summary["mode"] == "recall"


def test_balanced_mode_is_default_and_invalid_mode_fallback(pipeline_context) -> None:
    default_config = RowConfidenceGateConfig.from_context(pipeline_context)
    assert default_config.mode == "balanced"
    assert default_config.min_confidence == 0.40
    assert default_config.suppress_weak_when_identity_present is True
    assert default_config.identity_present_min_confidence == 0.45

    pipeline_context.config["v5_row_confidence_gate"] = {"mode": "diagnostic"}

    assert RowConfidenceGateConfig.from_context(pipeline_context).mode == "balanced"


def test_default_balanced_threshold_prefers_identity_when_present(
    pipeline_context,
) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(10, 10, 100, 100, color="red", price_card="1499.99"),
        _bbox_row(
            110,
            10,
            200,
            100,
            color="red",
            price_card="1499.99",
            price_default="1899.99",
            discount_amount="-21%",
        ),
        _bbox_row(210, 10, 300, 100, product_name="Wine", barcode="8051070512049"),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["identity_present_gate_active"] is True
    assert outcome.output_summary["identity_present_suppressed_rows"] == 1
    assert outcome.output_summary["kept_rows"] == 1
    assert outcome.output_summary["suppressed_rows"] == 2
    assert pipeline_context.csv_rows[0]["barcode"] == "8051070512049"


def test_balanced_mode_suppresses_low_evidence_localized_rows(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "min_confidence": 0.30,
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(10, 10, 100, 100),  # localized only -> diagnostic, not CSV
        _bbox_row(110, 10, 200, 100, color="red"),  # color+localized only
        _bbox_row(210, 10, 300, 100, color="red", price_card="1499.99"),
        _bbox_row(
            310,
            10,
            400,
            100,
            color="red",
            price_card="1499.99",
            price_default="1899.99",
        ),
        _bbox_row(410, 10, 500, 100, barcode="8051070512049"),
        {"barcode": "8051070512049"},  # no bbox -> not a row in final CSV
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert len(pipeline_context.csv_rows) == 2
    assert any(row.get("barcode", "") == "8051070512049" for row in pipeline_context.csv_rows)
    assert any(row.get("price_default", "") == "1899.99" for row in pipeline_context.csv_rows)
    assert outcome.output_summary["mode"] == "balanced"
    assert outcome.output_summary["input_rows"] == 6
    assert outcome.output_summary["kept_rows"] == 2
    assert outcome.output_summary["suppressed_rows"] == 4


def test_balanced_mode_keeps_confident_evidence_rows(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "min_confidence": 0.30,
        "suppress_weak_when_identity_present": False,
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(10, 10, 100, 100, product_name="Wine", barcode="8051070512049"),
        _bbox_row(110, 10, 200, 100, barcode="3500610117022"),
        _bbox_row(210, 10, 300, 100, qr_code_barcode="4690491122587"),
        _bbox_row(310, 10, 400, 100, product_name="Wine", color="red"),
        _bbox_row(410, 10, 500, 100, price_card="1499.99", price_default="1899.99"),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_rows"] == 5
    assert outcome.output_summary["suppressed_rows"] == 0


def test_balanced_mode_raises_price_only_bar_when_confirmed_identity_exists(
    pipeline_context,
) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(
            10,
            10,
            100,
            100,
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        ),
        _bbox_row(
            110,
            10,
            200,
            100,
            product_name="Known Wine",
            barcode="3500610117022",
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        ),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["identity_present_gate_active"] is True
    assert outcome.output_summary["identity_present_suppressed_rows"] == 1
    assert len(pipeline_context.csv_rows) == 1
    assert pipeline_context.csv_rows[0]["product_name"] == "Known Wine"


def test_balanced_mode_keeps_price_rows_when_no_confirmed_identity_exists(
    pipeline_context,
) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(
            10,
            10,
            100,
            100,
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        ),
        _bbox_row(
            110,
            10,
            200,
            100,
            barcode="3500610117022",
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        ),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["identity_present_gate_active"] is False
    assert outcome.output_summary["identity_present_suppressed_rows"] == 0
    assert len(pipeline_context.csv_rows) == 2


def test_identity_present_gate_keeps_match_key_only_rows(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(
            10,
            10,
            100,
            100,
            barcode="3500610117022",
        ),
        _bbox_row(
            110,
            10,
            200,
            100,
            product_name="Known Wine",
            barcode="8051070512049",
        ),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["identity_present_gate_active"] is True
    assert outcome.output_summary["identity_present_suppressed_rows"] == 0
    assert len(pipeline_context.csv_rows) == 2


def test_balanced_mode_suppresses_inverted_price_pair_without_identity(
    pipeline_context,
) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "spatial_dedup_iou": 0.0,
        "suppress_weak_when_identity_present": False,
    }
    pipeline_context.csv_rows = [
        _bbox_row(
            10,
            10,
            100,
            100,
            color="red",
            price_card="2749.99",
            price_default="2526.00",
            discount_amount="-37%",
        ),
        _bbox_row(
            110,
            10,
            200,
            100,
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        ),
        _bbox_row(
            210,
            10,
            300,
            100,
            product_name="Known Wine",
            barcode="3500610117022",
            color="red",
            price_card="2749.99",
            price_default="2526.00",
            discount_amount="-37%",
        ),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_rows"] == 2
    assert outcome.output_summary["suppressed_rows"] == 1
    assert all(row.get("x_min") != "10" for row in pipeline_context.csv_rows)
    assert any(row.get("product_name") == "Known Wine" for row in pipeline_context.csv_rows)


def test_balanced_mode_thresholds_single_weak_signals(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "min_confidence": 0.30,
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(10, 10, 100, 100, product_name="Likely catalog guess"),
        _bbox_row(
            110,
            10,
            200,
            100,
            catalog_match_status="catalog_guess",
            catalog_guess_name="Likely Wine",
        ),
        _bbox_row(210, 10, 300, 100, price_card="1499.99"),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert pipeline_context.csv_rows == []
    assert outcome.output_summary["kept_rows"] == 0
    assert outcome.output_summary["suppressed_rows"] == 3


def test_balanced_mode_keeps_high_confidence_sku(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "min_confidence": 0.30,
        "spatial_dedup_iou": 0.0,
    }
    pipeline_context.csv_rows = [
        _bbox_row(
            10,
            10,
            100,
            100,
            id_sku="270102701074",
            id_sku_confidence="0.94",
        ),
        _bbox_row(
            110,
            10,
            200,
            100,
            code="01_026015 - 026016",
        ),
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    assert outcome.output_summary["kept_rows"] == 1
    assert pipeline_context.csv_rows[0]["id_sku"] == "270102701074"


def test_spatial_dedup_collapses_same_tag_keeps_higher_confidence() -> None:
    weak = _bbox_row(100, 100, 300, 300, color="red")
    strong = _bbox_row(
        104,
        102,
        305,
        301,
        product_name="Wine X",
        barcode="3500610117022",
        price_card="1899.99",
    )
    kept, groups = _spatial_dedup([weak, strong], 0.6)

    assert len(kept) == 1
    assert kept[0]["barcode"] == "3500610117022"
    assert groups == 1


def test_spatial_dedup_preserves_distinct_identified_products() -> None:
    a = _bbox_row(100, 100, 300, 300, product_name="Wine A", barcode="3500610117022")
    b = _bbox_row(102, 101, 301, 299, product_name="Wine B", barcode="4690491122587")
    kept, _ = _spatial_dedup([a, b], 0.6)

    assert len(kept) == 2  # different products at one spot -> never merged


def test_spatial_dedup_preserves_spatially_separate_rows() -> None:
    a = _bbox_row(100, 100, 200, 200, color="red")
    b = _bbox_row(900, 900, 1000, 1000, color="red")
    kept, groups = _spatial_dedup([a, b], 0.6)

    assert len(kept) == 2
    assert groups == 0


def test_confidence_is_monotonic_and_bounded() -> None:
    bare = _row_confidence(_bbox_row(0, 0, 10, 10, color="red"))
    named = _row_confidence(_bbox_row(0, 0, 10, 10, color="red", product_name="W"))
    identified = _row_confidence(
        _bbox_row(
            0,
            0,
            10,
            10,
            color="red",
            product_name="W",
            barcode="3500610117022",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        )
    )

    assert 0.0 <= bare["confidence"] <= named["confidence"]
    assert named["confidence"] < identified["confidence"] <= 1.0
    assert "identity" in identified["evidence"]
    assert identified["tier"] == "high"
    assert bare["tier"] == "low"


def test_confidence_does_not_reward_inverted_price_default_or_discount() -> None:
    valid = _row_confidence(
        _bbox_row(
            0,
            0,
            10,
            10,
            color="red",
            price_card="1899.99",
            price_default="2631.00",
            discount_amount="-27%",
        )
    )
    inverted = _row_confidence(
        _bbox_row(
            0,
            0,
            10,
            10,
            color="red",
            price_card="2749.99",
            price_default="2526.00",
            discount_amount="-37%",
        )
    )

    assert "price_default" in valid["evidence"]
    assert "discount" in valid["evidence"]
    assert "price_default" not in inverted["evidence"]
    assert "discount" not in inverted["evidence"]
    assert inverted["confidence"] < 0.40


def test_gate_writes_confidence_side_car(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "balanced",
        "min_confidence": 0.30,
    }
    pipeline_context.csv_rows = [
        _bbox_row(1, 1, 5, 5),
        _bbox_row(
            6,
            6,
            9,
            9,
            catalog_match_status="catalog_guess",
            catalog_guess_name="Likely Wine",
        ),
        _bbox_row(
            10,
            10,
            100,
            100,
            product_name="Wine X",
            barcode="3500610117022",
        )
    ]

    outcome = V5RowConfidenceGateStage().run(pipeline_context)

    key = outcome.output_summary["row_confidence_key"]
    assert key.endswith("debug/row_confidence.json")
    stored = pipeline_context.artifact_writer.storage.objects
    assert key in stored
    assert outcome.output_summary["identity_rows"] == 1
    payload = stored[key][0].decode("utf-8")
    assert '"candidates"' in payload
    assert '"suppressed_reason": "no_evidence"' in payload
    assert '"catalog_guess_name": "Likely Wine"' in payload
