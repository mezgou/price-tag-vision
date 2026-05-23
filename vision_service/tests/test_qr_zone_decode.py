from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.pipelines.price_tag_v4.stages import qr_zone_decode as stage_module
from app.pipelines.price_tag_v4.stages.qr_zone_decode import (
    QrZoneDecodeStage,
    _cap_crops_per_track,
    _filter_variants,
    _has_code_evidence,
    _qr_zone_decoder_runners,
    _targeted_qr_zone_variants,
    qr_zones,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality, DecodedSymbol


def test_qr_zones_include_targeted_right_side_regions_with_white_border() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    zones = dict(qr_zones(image))

    assert {"upper_right", "right_mid", "qr_tight"}.issubset(zones)
    assert zones["qr_tight"][0, 0].tolist() == [255, 255, 255]


def test_qr_zones_can_be_limited_to_enabled_names() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    zones = dict(qr_zones(image, enabled_zones=("right_mid",)))

    assert list(zones) == ["right_mid"]


def test_targeted_qr_zone_variants_include_high_res_otsu() -> None:
    zone = np.zeros((20, 30, 3), dtype=np.uint8)
    zone[4:16, 8:22] = 255

    variants = _targeted_qr_zone_variants(zone)

    assert {"x4_gray", "x4_sharp", "x4_otsu"}.issubset(variants)
    assert variants["x4_otsu"].image.shape == (80, 120)
    assert variants["x4_otsu"].scale_x == 4.0


def test_qr_zone_variants_can_be_limited_to_enabled_names() -> None:
    zone = np.zeros((20, 30, 3), dtype=np.uint8)

    variants = _filter_variants(
        _targeted_qr_zone_variants(zone),
        enabled_variants=("x4_otsu",),
    )

    assert list(variants) == ["x4_otsu"]


def test_qr_zone_runner_prefers_qr_only_zxingcpp() -> None:
    runners = _qr_zone_decoder_runners(
        [
            ("zxingcpp_qr_only", lambda variant: []),
            ("zxingcpp", lambda variant: []),
            ("pyzbar", lambda variant: []),
        ]
    )

    assert [name for name, _ in runners] == ["zxingcpp_qr_only"]


def test_qr_zone_decode_skips_only_crops_from_tracks_with_code_evidence(
    pipeline_context,
    monkeypatch,
) -> None:
    crop = _crop(track_id="track_1")
    pipeline_context.crop_candidates = [crop]
    pipeline_context.config["barcode_qr_zone_decode"] = {
        "enabled": True,
        "skip_if_code_evidence_present": True,
    }
    monkeypatch.setattr(
        stage_module,
        "_resolve_decoder_runners",
        lambda *, config, warnings: [("fake", lambda variant: [])],
    )
    pipeline_context.decoded_symbols = [
        DecodedSymbol(
            symbol_id="sym_1",
            crop_id="crop_1",
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=0,
            symbol_type="qr",
            decoder="zxingcpp",
            variant="resized_x3",
            payload="barcode=8051070512049",
            confidence=0.9,
            attributes={"track_id": "track_1"},
        )
    ]

    outcome = QrZoneDecodeStage().run(pipeline_context)

    assert outcome.output_summary["crops_processed"] == 0
    assert outcome.output_summary["crops_skipped_by_code_evidence"] == 1
    assert outcome.output_summary["tracks_skipped_by_code_evidence"] == 1


def test_qr_zone_code_evidence_requires_qr_or_valid_ean() -> None:
    invalid_ean = DecodedSymbol(
        symbol_id="sym_1",
        crop_id="crop_1",
        detection_id="det_1",
        frame_index=0,
        timestamp_ms=0,
        symbol_type="barcode",
        decoder="zxingcpp",
        variant="original",
        payload="8051070512040",
        confidence=0.9,
    )
    qr_without_barcode = DecodedSymbol(
        symbol_id="sym_2",
        crop_id="crop_2",
        detection_id="det_2",
        frame_index=0,
        timestamp_ms=0,
        symbol_type="qr",
        decoder="zxingcpp",
        variant="original",
        payload="p1=123.00",
        confidence=0.9,
    )

    assert _has_code_evidence([invalid_ean]) is False
    assert _has_code_evidence([qr_without_barcode]) is True


def test_qr_zone_cap_leaves_uncoded_tracks_eligible() -> None:
    crops = [
        SimpleNamespace(attributes={"track_id": "coded"}, detection_id="det_1"),
        SimpleNamespace(attributes={"track_id": "uncoded"}, detection_id="det_2"),
    ]

    selected = _cap_crops_per_track(
        crops,
        max_total=10,
        max_per_track=0,
        skip_track_ids={"coded"},
    )

    assert [crop.attributes["track_id"] for crop in selected] == ["uncoded"]


def _crop(track_id: str = "track_1") -> CropCandidate:
    bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=60)
    return CropCandidate(
        crop_id="crop_1",
        detection_id="det_1",
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
            score=0.9,
        ),
        source="test",
        attributes={"track_id": track_id},
    )
