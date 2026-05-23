from __future__ import annotations

from app.pipelines.price_tag_v2.stages.row_fusion import RowFusionStage
from app.pipelines.price_tag_v5.catalog import CatalogMatch
from app.pipelines.price_tag_v5.stages.block_ocr_catalog import (
    _barcode_values_from_symbol,
    _best_barcode_hint,
    _catalog_identity_fields,
    _visual_barcode_identity_fields,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality, DetectionCandidate


def test_catalog_match_with_reliable_key_promotes_product_name() -> None:
    match = CatalogMatch(
        product_name="Wine A",
        barcode="8051070512049",
        score=12.345,
        margin=7.0,
        accepted=True,
        candidate_codes=("8051070512049",),
        status="confirmed",
    )

    fields = _catalog_identity_fields(match, barcode_hint="8051070512049")

    assert fields["product_name"] == "Wine A"
    assert fields["barcode"] == "8051070512049"
    assert fields["qr_code_barcode"] == "8051070512049"
    assert fields["catalog_match_status"] == "confirmed"
    assert fields["catalog_key_source"] == "visual_barcode_or_qr"


def test_catalog_match_with_only_ocr_hint_stays_guess() -> None:
    match = CatalogMatch(
        product_name="Wine A",
        barcode="8051070512049",
        score=12.345,
        margin=7.0,
        accepted=True,
        candidate_codes=("8051070512049",),
        status="confirmed",
    )

    fields = _catalog_identity_fields(
        match,
        barcode_hint="8051070512049",
        reliable_barcode_hint="",
    )

    assert "product_name" not in fields
    assert "barcode" not in fields
    assert fields["catalog_match_status"] == "catalog_guess"
    assert fields["catalog_key_source"] == "catalog_inferred"


def test_qr_payload_barcode_is_reliable_catalog_hint() -> None:
    assert _barcode_values_from_symbol("barcode=8051070512049;p1=1299.99") == [
        "8051070512049"
    ]
    assert _best_barcode_hint(["8051070512049"], fallback="") == "8051070512049"


def test_visual_barcode_identity_fields_are_confirmed() -> None:
    fields = _visual_barcode_identity_fields("8051070512049", "Wine A")

    assert fields["product_name"] == "Wine A"
    assert fields["barcode"] == "8051070512049"
    assert fields["qr_code_barcode"] == "8051070512049"
    assert fields["catalog_match_status"] == "confirmed"
    assert fields["catalog_key_source"] == "visual_barcode_or_qr"


def test_catalog_match_without_reliable_key_stays_guess() -> None:
    match = CatalogMatch(
        product_name="Wine A",
        barcode="8051070512049",
        score=12.345,
        margin=7.0,
        accepted=True,
        candidate_codes=("8051070512049",),
        status="confirmed",
    )

    fields = _catalog_identity_fields(match, barcode_hint="")

    assert "product_name" not in fields
    assert "barcode" not in fields
    assert "qr_code_barcode" not in fields
    assert fields["catalog_match_status"] == "catalog_guess"
    assert fields["catalog_resolver_status"] == "confirmed"
    assert fields["catalog_key_source"] == "catalog_inferred"
    assert fields["catalog_withheld_reason"] == "no_reliable_barcode"
    assert fields["catalog_guess_name"] == "Wine A"
    assert fields["catalog_guess_barcode"] == "8051070512049"


def test_best_effort_catalog_match_stays_guess() -> None:
    match = CatalogMatch(
        product_name="",
        barcode="",
        score=4.0,
        margin=1.0,
        accepted=False,
        candidate_codes=("8051070512049",),
        catalog_guess_name="Likely Wine",
        status="catalog_guess",
    )

    fields = _catalog_identity_fields(match, barcode_hint="")

    assert "product_name" not in fields
    assert "barcode" not in fields
    assert fields["catalog_guess_name"] == "Likely Wine"
    assert fields["catalog_match_status"] == "catalog_guess"


def test_row_fusion_preserves_catalog_guess_metadata(pipeline_context) -> None:
    bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120)
    pipeline_context.detections = [
        DetectionCandidate(
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=100,
            label="price_tag_candidate",
            bbox=bbox,
            confidence=0.9,
            source="test",
            attributes={"track_id": "track_1"},
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="crop_1",
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=100,
            bbox=bbox,
            padded_bbox=bbox,
            crop_key="",
            width=100,
            height=100,
            quality=CropQuality(
                sharpness=1.0,
                brightness=100.0,
                contrast=20.0,
                glare_ratio=0.0,
                area_ratio=0.1,
                score=0.8,
            ),
            source="test",
            attributes={
                "ocr": {
                    "confidence": 0.8,
                    "fields": {
                        "catalog_match_status": "catalog_guess",
                        "catalog_resolver_status": "confirmed",
                        "catalog_withheld_reason": "no_reliable_barcode",
                        "catalog_guess_name": "Likely Wine",
                        "catalog_guess_barcode": "8051070512049",
                        "price_card": "1299.99",
                    },
                }
            },
        )
    ]

    RowFusionStage().run(pipeline_context)

    row = pipeline_context.csv_rows[0]
    assert row["product_name"] == ""
    assert row["barcode"] == ""
    assert row["catalog_match_status"] == "catalog_guess"
    assert row["catalog_resolver_status"] == "confirmed"
    assert row["catalog_guess_name"] == "Likely Wine"
    assert row["catalog_guess_barcode"] == "8051070512049"
