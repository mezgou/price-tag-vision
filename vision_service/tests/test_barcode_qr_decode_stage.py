from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.pipelines.base import SampledFrameMetadata
from app.pipelines.price_tag_cpu_v1.stages import barcode_qr_decode as decode_stage_module
from app.pipelines.price_tag_cpu_v1.stages.barcode_qr_decode import (
    BarcodeQrDecodeStage,
    BarcodeQrDecodeConfig,
    OPTIONAL_ZXINGCPP_DECODER,
    OPTIONAL_ZXINGCPP_QR_ONLY_DECODER,
    _cap_crops_per_track,
    _decode_with_optional_zxingcpp_qr_only,
    _resolve_decoder_runners,
)
from app.schemas.detections import BoundingBox, CropCandidate, CropQuality
from app.utils.decoding import DecodeVariantImage


def test_barcode_qr_decode_stage_handles_empty_crops(
    pipeline_context,
) -> None:
    pipeline_context.config["barcode_qr_decode"] = {"enabled": True}
    pipeline_context.crop_candidates = []

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decode_attempts_count"] == 0
    assert outcome.output_summary["decoded_symbols_count"] == 0
    assert pipeline_context.decode_attempts == []
    assert pipeline_context.decoded_symbols == []


def test_barcode_qr_decode_stage_handles_crop_without_symbols(
    pipeline_context,
    tmp_path: Path,
) -> None:
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (40, 70), (160, 130), (245, 245, 245), thickness=-1)
    frame_path = tmp_path / "runtime" / "frame_no_qr.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["barcode_qr_decode"] = {
        "enabled": True,
        "max_crops": 10,
        "min_crop_quality_score": 0.0,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=0,
            timestamp_ms=0,
            original_width=200,
            original_height=200,
            processed_width=200,
            processed_height=200,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="crop_1",
            detection_id="det_1",
            frame_index=0,
            timestamp_ms=0,
            bbox=BoundingBox(x_min=40, y_min=70, x_max=160, y_max=130),
            padded_bbox=BoundingBox(x_min=40, y_min=70, x_max=160, y_max=130),
            crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
            width=120,
            height=60,
            quality=CropQuality(
                sharpness=10.0,
                brightness=220.0,
                contrast=12.0,
                glare_ratio=0.0,
                area_ratio=0.18,
                score=0.6,
            ),
            source="crop_extraction_v1",
            attributes={"track_id": "track_qr"},
        )
    ]

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decode_attempts_count"] > 0
    assert outcome.output_summary["decoded_symbols_count"] == 0
    assert len(pipeline_context.decode_attempts) > 0
    assert pipeline_context.decoded_symbols == []


def test_barcode_qr_decode_stage_decodes_synthetic_qr_image(
    pipeline_context,
    tmp_path: Path,
) -> None:
    qr_payload = "https://example.test/price-tag-vision"
    qr_image = _build_qr_image(qr_payload, target_size=120)

    frame = np.full((220, 220, 3), 255, dtype=np.uint8)
    frame[50:170, 50:170] = qr_image
    frame_path = tmp_path / "runtime" / "frame_qr.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(frame_path), frame)

    pipeline_context.config["barcode_qr_decode"] = {
        "enabled": True,
        "max_crops": 10,
        "min_crop_quality_score": 0.0,
        "stop_after_first_success_per_crop": False,
    }
    pipeline_context.sampled_frames = [
        SampledFrameMetadata(
            frame_index=5,
            timestamp_ms=1000,
            original_width=220,
            original_height=220,
            processed_width=220,
            processed_height=220,
            orientation_applied="none",
            local_frame_path=frame_path,
        )
    ]
    pipeline_context.crop_candidates = [
        CropCandidate(
            crop_id="crop_qr",
            detection_id="det_qr",
            frame_index=5,
            timestamp_ms=1000,
            bbox=BoundingBox(x_min=50, y_min=50, x_max=170, y_max=170),
            padded_bbox=BoundingBox(x_min=45, y_min=45, x_max=175, y_max=175),
            crop_key="outputs/job-123/debug/crops/frame_000001_det_000001.jpg",
            width=130,
            height=130,
            quality=CropQuality(
                sharpness=1000.0,
                brightness=180.0,
                contrast=90.0,
                glare_ratio=0.0,
                area_ratio=0.35,
                score=0.95,
            ),
            source="crop_extraction_v1",
            attributes={"track_id": "track_qr"},
        )
    ]

    outcome = BarcodeQrDecodeStage().run(pipeline_context)

    assert outcome.output_summary["decoded_symbols_count"] >= 1
    assert outcome.output_summary["decoded_symbols_by_type"]["qr"] >= 1
    assert len(pipeline_context.decode_attempts) > 0
    assert len(pipeline_context.decoded_symbols) >= 1
    assert any(symbol.payload == qr_payload for symbol in pipeline_context.decoded_symbols)
    assert all(symbol.symbol_type == "qr" for symbol in pipeline_context.decoded_symbols)
    assert {symbol.attributes.get("track_id") for symbol in pipeline_context.decoded_symbols} == {
        "track_qr"
    }
    assert len({symbol.payload for symbol in pipeline_context.decoded_symbols}) == 1


def test_barcode_qr_decode_caps_crops_per_track() -> None:
    crops = [
        _crop(f"crop_{i}", track_id="track_a", score=1.0 - i * 0.01)
        for i in range(4)
    ] + [
        _crop(f"crop_b_{i}", track_id="track_b", score=0.8 - i * 0.01)
        for i in range(3)
    ]

    selected = _cap_crops_per_track(crops, max_total=10, max_per_track=2)

    assert [crop.crop_id for crop in selected] == [
        "crop_0",
        "crop_1",
        "crop_b_0",
        "crop_b_1",
    ]


def test_zxingcpp_qr_only_runner_is_ordered_before_generic_zxingcpp(
    pipeline_context,
    monkeypatch,
) -> None:
    pipeline_context.config["barcode_qr_decode"] = {
        "enabled": True,
        "decoders": {
            "opencv_qr_detector": {"enabled": False},
            "optional_zxingcpp_qr_only": {"enabled": True},
            "optional_zxingcpp": {"enabled": True},
            "optional_pyzbar": {"enabled": False},
            "optional_aruco": {"enabled": False},
        },
    }
    config = BarcodeQrDecodeConfig.from_context(pipeline_context)
    warnings: list[str] = []

    monkeypatch.setattr(
        decode_stage_module,
        "find_spec",
        lambda name: object() if name == "zxingcpp" else None,
    )

    runners = _resolve_decoder_runners(config=config, warnings=warnings)

    assert [name for name, _ in runners] == [
        OPTIONAL_ZXINGCPP_QR_ONLY_DECODER,
        OPTIONAL_ZXINGCPP_DECODER,
    ]
    assert warnings == []


def test_zxingcpp_qr_only_runner_requests_qr_format(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    qr_format = object()

    def read_barcodes(image: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        calls.append(kwargs)
        return [SimpleNamespace(text="qr-payload", format="QRCode")]

    fake_zxingcpp = SimpleNamespace(
        BarcodeFormat=SimpleNamespace(QRCode=qr_format),
        read_barcodes=read_barcodes,
    )
    monkeypatch.setitem(sys.modules, "zxingcpp", fake_zxingcpp)
    variant = DecodeVariantImage(
        name="resized_x3",
        image=np.zeros((16, 16), dtype=np.uint8),
        scale_x=3.0,
        scale_y=3.0,
    )

    hits = _decode_with_optional_zxingcpp_qr_only(variant)

    assert calls == [
        {
            "try_rotate": True,
            "try_downscale": True,
            "try_invert": True,
            "formats": qr_format,
        }
    ]
    assert len(hits) == 1
    assert hits[0].payload == "qr-payload"
    assert hits[0].symbol_type == "qr"
    assert hits[0].attributes == {"format": "qrcode", "qr_only": True}


def test_zxingcpp_qr_only_runner_rejects_non_qr_formats(monkeypatch) -> None:
    def read_barcodes(image: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(text="4600000000000", format="EAN13")]

    fake_zxingcpp = SimpleNamespace(
        BarcodeFormat=SimpleNamespace(QRCode=object()),
        read_barcodes=read_barcodes,
    )
    monkeypatch.setitem(sys.modules, "zxingcpp", fake_zxingcpp)
    variant = DecodeVariantImage(
        name="original",
        image=np.zeros((16, 16), dtype=np.uint8),
    )

    assert _decode_with_optional_zxingcpp_qr_only(variant) == []


def _build_qr_image(payload: str, *, target_size: int) -> np.ndarray:
    params = cv2.QRCodeEncoder_Params()
    encoder = cv2.QRCodeEncoder_create(params)
    qr = encoder.encode(payload)
    qr = cv2.resize(qr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def _crop(crop_id: str, *, track_id: str, score: float) -> CropCandidate:
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    return CropCandidate(
        crop_id=crop_id,
        detection_id=f"det_{crop_id}",
        frame_index=0,
        timestamp_ms=0,
        bbox=bbox,
        padded_bbox=bbox,
        crop_key="",
        width=10,
        height=10,
        quality=CropQuality(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            glare_ratio=0.0,
            area_ratio=0.0,
            score=score,
        ),
        source="test",
        attributes={"track_id": track_id},
    )
