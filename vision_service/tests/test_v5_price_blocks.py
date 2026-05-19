"""Unit coverage for the v5 price reconciliation + colour mapping.

Block sets mirror the real close-clip OCR (SAMMARCO / BURCHINO) where the
old tallest-block heuristic mis-picked the struck default integer as the
card price (price_card + price4_qr both wrong on most rows).
"""
from __future__ import annotations

from app.pipelines.price_tag_v5.stages.block_ocr_catalog import (
    _COLOR_TO_EN,
    V5BlockOcrCatalogStage as S,
)


def _b(text: str, rel_y: float = 0.5, rel_h: float = 0.1) -> dict:
    return {"text": text, "conf": 0.9, "rel_y": rel_y, "rel_h": rel_h}


def test_card_is_smaller_price_with_99_kopecks() -> None:
    # SAMMARCO: struck default 1894(.73), big card 1299(.99), disc -31%
    blocks = [_b("1894", 0.30, 0.06), _b("1299", 0.60, 0.14),
              _b("99", 0.58, 0.05), _b("-31%", 0.55, 0.05)]
    out = S._prices_from_blocks(blocks)
    assert out["price_card"] == "1299.99"
    assert out["price_default"] == "1894.00"
    assert out["discount_amount"] == "-31%"


def test_card_not_the_tallest_block() -> None:
    # default integer rendered TALLER than the card price -> old code failed
    blocks = [_b("2894", 0.30, 0.20), _b("1999", 0.62, 0.10),
              _b("99", 0.60, 0.04)]
    out = S._prices_from_blocks(blocks)
    assert out["price_card"] == "1999.99"
    assert out["price_default"] == "2894.00"


def test_single_price_treated_as_card() -> None:
    out = S._prices_from_blocks([_b("1299", 0.6, 0.14), _b("99", 0.58, 0.04)])
    assert out["price_card"] == "1299.99"
    assert "price_default" not in out


def test_volume_token_not_a_price() -> None:
    out = S._prices_from_blocks([_b("0.75L", 0.2, 0.05), _b("1199", 0.6, 0.14),
                                 _b("99", 0.58, 0.04)])
    assert out["price_card"] == "1199.99"


def test_kopeck_prefers_99_over_other_two_digit() -> None:
    out = S._prices_from_blocks([_b("1894", 0.3), _b("1299", 0.6, 0.14),
                                 _b("31", 0.4), _b("99", 0.58)])
    assert out["price_card"].endswith(".99")


def test_no_price_blocks_yields_nothing() -> None:
    assert S._prices_from_blocks([_b("abc", 0.5), _b("-31%", 0.5)]) == {
        "discount_amount": "-31%"
    }


def test_colour_map_to_english() -> None:
    assert _COLOR_TO_EN["красный"] == "red"
    assert _COLOR_TO_EN["жёлтый"] == "yellow"
    assert _COLOR_TO_EN["белый"] == "white"
