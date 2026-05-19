"""Unit coverage for the strictly-non-negative cross-field derivations
added to RowFusion (discount/price2_qr/additional_info + нет backfill).

Values are taken straight from the labelled GT (26_12-20 / 43_15) so a
regression here means a real measured-accuracy regression.
"""
from __future__ import annotations

import pytest

from app.pipelines.price_tag_v2.stages.row_fusion import (
    DEFAULT_ABSENT_FIELDS,
    _derive_cross_fields,
    _sweetness_from_name,
    _to_price,
)


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
