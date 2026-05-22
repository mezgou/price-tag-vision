from __future__ import annotations

import numpy as np

from app.pipelines.price_tag_v4.stages.qr_zone_decode import (
    _targeted_qr_zone_variants,
    qr_zones,
)


def test_qr_zones_include_targeted_right_side_regions_with_white_border() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    zones = dict(qr_zones(image))

    assert {"upper_right", "right_mid", "qr_tight"}.issubset(zones)
    assert zones["qr_tight"][0, 0].tolist() == [255, 255, 255]


def test_targeted_qr_zone_variants_include_high_res_otsu() -> None:
    zone = np.zeros((20, 30, 3), dtype=np.uint8)
    zone[4:16, 8:22] = 255

    variants = _targeted_qr_zone_variants(zone)

    assert {"x4_gray", "x4_sharp", "x4_otsu"}.issubset(variants)
    assert variants["x4_otsu"].image.shape == (80, 120)
    assert variants["x4_otsu"].scale_x == 4.0
