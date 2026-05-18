from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_cpu_v1.pipeline import (
    PriceTagCpuV1Pipeline,
    _load_yaml_config,
)
from app.pipelines.price_tag_cpu_v1.stages import (
    CropExtractionStage,
    CsvWriterStage,
    DebugManifestStage,
    FrameMetadataStage,
    FrameSamplingStage,
    PreviewWriterStage,
)
from app.pipelines.price_tag_v2.stages import (
    RowFusionStage,
    TopKCropSelectionStage,
    YoloByteTrackStage,
)
from app.pipelines.price_tag_v5.stages import V5BlockOcrCatalogStage


class PriceTagV5Pipeline(PriceTagCpuV1Pipeline):
    """Catalog-anchored, GT-free recognition.

    Reuses v2's proven detector/ByteTrack/crop/row-fusion; replaces the slow
    fixed-zone OCR with one fast full-crop block OCR + db_hack catalog
    identity. v2/v3/v4 are untouched.
    """

    def __init__(self) -> None:
        self._config_path = Path(__file__).with_name("config.yaml")
        self._base_config = _load_yaml_config(self._config_path)
        pipeline_config = self._base_config.get("pipeline", {})
        self.name = str(pipeline_config.get("name", "price_tag_v5"))
        self.default_version = str(pipeline_config.get("version", "0.1.0"))
        self._stages = [
            FrameMetadataStage(),
            FrameSamplingStage(),
            YoloByteTrackStage(),
            CropExtractionStage(),
            TopKCropSelectionStage(),
            V5BlockOcrCatalogStage(),
            RowFusionStage(),
            CsvWriterStage(),
            PreviewWriterStage(),
            DebugManifestStage(),
        ]
