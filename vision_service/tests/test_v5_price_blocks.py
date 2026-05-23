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


def _b(
    text: str,
    rel_y: float = 0.5,
    rel_h: float = 0.1,
    rel_x: float = 0.45,
    rel_w: float = 0.15,
) -> dict:
    return {
        "text": text,
        "conf": 0.9,
        "rel_y": rel_y,
        "rel_h": rel_h,
        "rel_x": rel_x,
        "rel_w": rel_w,
        "rel_cx": rel_x + rel_w / 2.0,
    }


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


def test_default_kopecks_use_nearby_two_digit_ocr_block() -> None:
    out = S._prices_from_blocks(
        [
            _b("2631", 0.24, 0.08),
            _b("57", 0.25, 0.03),
            _b("1899", 0.63, 0.14),
            _b("99", 0.64, 0.04),
            _b("-27%", 0.55, 0.05),
        ]
    )
    assert out["price_default"] == "2631.57"
    assert out["price_card"] == "1899.99"


def test_default_kopecks_fall_back_to_zero_without_nearby_evidence() -> None:
    out = S._prices_from_blocks(
        [
            _b("2631", 0.24, 0.08),
            _b("1899", 0.63, 0.14),
            _b("99", 0.64, 0.04),
        ]
    )
    assert out["price_default"] == "2631.00"


def test_price_pair_prefers_tag_geometry_over_larger_neighbor_price() -> None:
    blocks = [
        _b("373", 0.22, 0.08, 0.42, 0.14),
        _b("69", 0.23, 0.03, 0.58, 0.06),
        _b("299", 0.58, 0.14, 0.39, 0.16),
        _b("99", 0.59, 0.04, 0.57, 0.07),
        _b("599", 0.78, 0.08, 0.42, 0.14),
        _b("50", 0.79, 0.03, 0.58, 0.06),
        _b("469", 0.91, 0.14, 0.39, 0.16),
        _b("99", 0.92, 0.04, 0.57, 0.07),
    ]

    out = S._prices_from_blocks(blocks)

    assert out["price_default"] == "373.69"
    assert out["price_card"] == "299.99"


def test_weight_token_is_not_a_price() -> None:
    out = S._prices_from_blocks(
        [_b("275г", 0.20), _b("305", 0.30), _b("244", 0.60), _b("99", 0.61)]
    )

    assert out["price_default"] == "305.00"
    assert out["price_card"] == "244.99"


def test_default_price_trims_trailing_ocr_zero_when_pair_becomes_plausible() -> None:
    out = S._prices_from_blocks(
        [
            _b("18940", 0.39, 0.08),
            _b("1299", 0.60, 0.14),
            _b("99", 0.58, 0.04),
            _b("-31%", 0.62, 0.05),
        ]
    )

    assert out["price_default"] == "1894.00"
    assert out["price_card"] == "1299.99"


def test_card_kopecks_ignore_left_noise_and_fall_back_to_99() -> None:
    out = S._prices_from_blocks(
        [_b("130", 0.60, 0.14, 0.45, 0.16), _b("16", 0.61, 0.04, 0.30, 0.06)]
    )

    assert out["price_card"] == "130.99"


def test_one_digit_discount_can_be_repaired_from_consistent_prices() -> None:
    out = S._prices_from_blocks(
        [
            _b("1578", 0.30, 0.08),
            _b("1199", 0.62, 0.14),
            _b("99", 0.63, 0.04),
            _b("4%", 0.58, 0.05),
        ]
    )

    assert out["discount_amount"] == "-24%"


def test_special_symbol_accepts_short_ko_bottom_read() -> None:
    out = S._prices_from_blocks([_b("KO", 0.84, 0.05)])

    assert out["special_symbols"] == "К"


def test_additional_info_from_latinized_sweetness_block() -> None:
    out = S._prices_from_blocks([_b("Cyxoe", 0.40, 0.08)])

    assert out["additional_info"] == "Сухое"


def test_no_price_blocks_yields_nothing() -> None:
    assert S._prices_from_blocks([_b("abc", 0.5), _b("-31%", 0.5)]) == {
        "discount_amount": "-31%"
    }


def test_colour_map_to_english() -> None:
    assert _COLOR_TO_EN["красный"] == "red"
    assert _COLOR_TO_EN["жёлтый"] == "yellow"
    assert _COLOR_TO_EN["белый"] == "white"
