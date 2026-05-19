from __future__ import annotations

from argparse import Namespace

from scripts.smoke_price_tag_pipeline import _build_request_config


def test_smoke_config_does_not_override_pipeline_defaults_by_default() -> None:
    args = Namespace(
        sample_fps=None,
        max_frames=None,
        max_candidates_per_frame=None,
        max_total_crops=None,
        max_crops_per_frame=None,
        config_json="",
        config_file=None,
    )

    assert _build_request_config(args) == {}


def test_smoke_config_includes_only_explicit_cli_overrides() -> None:
    args = Namespace(
        sample_fps=5.0,
        max_frames=None,
        max_candidates_per_frame=None,
        max_total_crops=900,
        max_crops_per_frame=None,
        config_json="",
        config_file=None,
    )

    assert _build_request_config(args) == {
        "frame_sampling": {"sample_fps": 5.0},
        "crop_extraction": {"max_total_crops": 900},
    }
