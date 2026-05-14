from __future__ import annotations

import numpy as np

from app.pipelines.price_tag_cpu_v1.orientation import (
    apply_orientation,
    resolve_orientation_mode,
)


def test_orientation_helper_supports_all_modes() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    rotated_none = apply_orientation(frame, "none")
    rotated_cw = apply_orientation(frame, "rotate_90_cw")
    rotated_ccw = apply_orientation(frame, "rotate_90_ccw")
    rotated_180 = apply_orientation(frame, "rotate_180")

    assert rotated_none.shape == (2, 3, 3)
    assert rotated_cw.shape == (3, 2, 3)
    assert rotated_ccw.shape == (3, 2, 3)
    assert rotated_180.shape == (2, 3, 3)
    assert np.array_equal(rotated_cw[0, 0], frame[1, 0])
    assert np.array_equal(rotated_ccw[0, 0], frame[0, 2])
    assert np.array_equal(rotated_180[0, 0], frame[1, 2])


def test_resolve_orientation_mode_warns_for_unknown_value() -> None:
    mode, warnings = resolve_orientation_mode("sideways")

    assert mode == "none"
    assert warnings == [
        "Unknown orientation_mode 'sideways'. Falling back to 'none'."
    ]
