"""Unit coverage for the strictly-non-negative cross-field derivations
added to RowFusion (discount/price2_qr/additional_info + нет backfill).

Values are taken straight from the labelled GT (26_12-20 / 43_15) so a
regression here means a real measured-accuracy regression.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from app.pipelines.price_tag_v2.stages.row_fusion import (
    DEFAULT_ABSENT_FIELDS,
    _add_ocr_votes,
    _choose_vote,
    _derive_cross_fields,
    _qr_price_should_override_visible,
    _stable_symbol_key,
    _sweetness_from_name,
    _to_price,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality, DecodedSymbol


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2345.99", 2345.99),
        ("2 345,99", 2345.99),
        ("3789,49", 3789.49),
        ("нет", None),
        ("", None),
        ("0", None),
        ("-12", None),
    ],
)
def test_to_price(raw: str, expected: float | None) -> None:
    assert _to_price(raw) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Вино CITRAN Бордо Суперьор кр. сух. (Франция) 0.75L", "Сухое"),
        ("Вино LE GRAND NOIR Пино Нуар кр. п./ сух. (Франция) 0.75L", "Полусухое"),
        ("Вино PRIMASOLE Примитиво Апулия кр. п/ сух.(Италия) 0.75L", "Полусухое"),
        ("Вино NOBILOMO Марцемино DOC кр. п/ сл. (Италия) 0.75L", "Полусладкое"),
        ("Игристое BRUT какое-то брют (Италия) 0.75L", "Брют"),
        ("Мед ПОТАПЫЧ Натуральный липовый (Россия) 500г", ""),
    ],
)
def test_sweetness_from_name(name: str, expected: str) -> None:
    assert _sweetness_from_name(name) == expected


def test_discount_derived_from_prices_truncates() -> None:
    # GT 26_12-20: default 3789.49 / card 2345.99 -> -38%
    row = {"price_default": "3789.49", "price_card": "2345.99",
           "discount_amount": "", "price2_qr": "", "product_name": ""}
    _derive_cross_fields(row)
    assert row["discount_amount"] == "-38%"


def test_discount_not_overwritten_when_present() -> None:
    row = {"price_default": "3789.49", "price_card": "2345.99",
           "discount_amount": "-37%", "price2_qr": "", "product_name": ""}
    _derive_cross_fields(row)
    assert row["discount_amount"] == "-37%"


def test_price2_qr_derived_matches_gt_pattern() -> None:
    # GT: 3789.49 -> 3599.99 ; 2421.05 -> 2299.99 ; 415.79 -> 394.99
    for default, expected in (("3789.49", "3599.99"),
                              ("2421.05", "2299.99"),
                              ("415.79", "394.99")):
        row = {"price_default": default, "price_card": "",
               "discount_amount": "x", "price2_qr": "", "product_name": ""}
        _derive_cross_fields(row)
        assert row["price2_qr"] == expected, default


def test_price2_qr_falls_back_to_price1_qr() -> None:
    row = {"price_default": "", "price1_qr": "3789.49", "price_card": "",
           "discount_amount": "x", "price2_qr": "", "product_name": ""}
    _derive_cross_fields(row)
    assert row["price2_qr"] == "3599.99"


def test_qr_price_overrides_integer_only_visible_price() -> None:
    assert _qr_price_should_override_visible(
        qr_field="price1_qr",
        qv="3157.89",
        vv="3157.00",
    )
    assert not _qr_price_should_override_visible(
        qr_field="price1_qr",
        qv="3157.89",
        vv="3158.00",
    )
    assert not _qr_price_should_override_visible(
        qr_field="price1_qr",
        qv="3157.89",
        vv="3157.49",
    )


def test_additional_info_derived_from_name_only_when_empty() -> None:
    row = {"price_default": "", "price_card": "", "discount_amount": "x",
           "price2_qr": "x",
           "product_name": "Вино X кр. п/ сух. (Италия) 0.75L",
           "additional_info": ""}
    _derive_cross_fields(row)
    assert row["additional_info"] == "Полусухое"

    row2 = dict(row, additional_info="Сухое")
    _derive_cross_fields(row2)
    assert row2["additional_info"] == "Сухое"


def test_no_derivation_without_sources() -> None:
    row = {"price_default": "", "price_card": "", "discount_amount": "",
           "price2_qr": "", "product_name": "", "additional_info": ""}
    _derive_cross_fields(row)
    assert row["discount_amount"] == ""
    assert row["price2_qr"] == ""
    assert row["additional_info"] == ""


def test_default_absent_now_covers_no_value_fields() -> None:
    for field in ("code", "special_symbols", "additional_info",
                  "price_discount", "wholesale_level_1_count",
                  "action_price_qr"):
        assert field in DEFAULT_ABSENT_FIELDS


def test_ocr_vote_uses_field_specific_confidence() -> None:
    votes = defaultdict(list)
    high = _crop("high", "270102701074", field_confidence=0.95, quality=0.5)
    low = _crop("low", "370204501518", field_confidence=0.10, quality=0.9)

    _add_ocr_votes(votes=votes, crops=[low, high])

    assert _choose_vote(field="id_sku", votes=votes["id_sku"]) == "270102701074"


def test_price_only_qr_payload_does_not_create_stable_barcode_key() -> None:
    symbol = DecodedSymbol(
        symbol_id="symbol_1",
        crop_id="crop_1",
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        symbol_type="qr",
        decoder="test",
        variant="test",
        payload="p1=1299.99;p4=999.99",
        confidence=0.9,
    )

    assert _stable_symbol_key(symbol) == ""


def _crop(
    crop_id: str,
    id_sku: str,
    *,
    field_confidence: float,
    quality: float,
) -> CropCandidate:
    bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60)
    return CropCandidate(
        crop_id=crop_id,
        detection_id=crop_id,
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
            score=quality,
        ),
        source="test",
        attributes={
            "ocr": {
                "confidence": 0.0,
                "field_confidences": {"id_sku": field_confidence},
                "fields": {"id_sku": id_sku},
            }
        },
    )
