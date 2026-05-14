from __future__ import annotations

import numpy as np

from app.utils.decoding import (
    build_decode_variants,
    build_payload_preview,
    normalize_decoded_payload,
)


def test_payload_normalization_helper_strips_whitespace_and_null_bytes() -> None:
    normalized = normalize_decoded_payload(" \x00hello\x00 world \n")

    assert normalized == "hello world"


def test_decode_variants_helper_builds_expected_variant_shapes() -> None:
    crop = np.zeros((40, 60, 3), dtype=np.uint8)
    variants = build_decode_variants(crop)

    assert variants["original"].image.shape == (40, 60, 3)
    assert variants["grayscale"].image.shape == (40, 60)
    assert variants["resized_x2"].image.shape == (80, 120, 3)
    assert variants["adaptive_threshold"].image.shape == (40, 60)


def test_payload_preview_limits_length_without_breaking_short_payloads() -> None:
    long_payload = "A" * 400

    preview = build_payload_preview(long_payload, max_length=20)

    assert len(preview) == 20
    assert preview.endswith("...")
