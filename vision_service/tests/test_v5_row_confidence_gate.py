from __future__ import annotations

from app.pipelines.price_tag_v5.stages.row_confidence_gate import (
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
    assert row_has_v5_evidence({"price_card": "1199.99"})


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


def test_gate_writes_confidence_side_car(pipeline_context) -> None:
    pipeline_context.config["v5_row_confidence_gate"] = {
        "enabled": True,
        "mode": "recall",
    }
    pipeline_context.csv_rows = [
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
