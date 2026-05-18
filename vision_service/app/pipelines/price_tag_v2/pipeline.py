from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_cpu_v1.pipeline import (
    PriceTagCpuV1Pipeline,
    _load_yaml_config,
)
from app.pipelines.price_tag_cpu_v1.stages import (
    BarcodeQrDecodeStage,
    CropExtractionStage,
    CsvWriterStage,
    DebugManifestStage,
    FrameMetadataStage,
    FrameSamplingStage,
    PreviewWriterStage,
)
from app.pipelines.price_tag_v2.stages import (
    FallbackHeuristicCandidateDetectionStage,
    RowFusionStage,
    TopKCropSelectionStage,
    YoloByteTrackStage,
    ZonalOcrStage,
)


class PriceTagV2Pipeline(PriceTagCpuV1Pipeline):
    def __init__(self) -> None:
        self._config_path = Path(__file__).with_name("config.yaml")
        self._base_config = _load_yaml_config(self._config_path)
        pipeline_config = self._base_config.get("pipeline", {})
        self.name = str(pipeline_config.get("name", "price_tag_v2"))
        self.default_version = str(pipeline_config.get("version", "0.2.0"))
        self._stages = [
            FrameMetadataStage(),
            FrameSamplingStage(),
            YoloByteTrackStage(),
            FallbackHeuristicCandidateDetectionStage(),
            CropExtractionStage(),
            TopKCropSelectionStage(),
            BarcodeQrDecodeStage(),
            ZonalOcrStage(),
            RowFusionStage(),
            CsvWriterStage(),
            PreviewWriterStage(),
            DebugManifestStage(),
        ]
