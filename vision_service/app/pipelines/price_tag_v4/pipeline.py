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
    PreviewWriterStage,
)
from app.pipelines.price_tag_v2.stages import (
    FallbackHeuristicCandidateDetectionStage,
    RowFusionStage,
    YoloByteTrackStage,
    ZonalOcrStage,
)
from app.pipelines.price_tag_v3.stages import (
    CharConsensusStage,
    FusedZonalDecodeStage,
    MultiViewFusionStage,
    TrackCropBankStage,
)
from app.pipelines.price_tag_v4.stages import (
    BarcodeBarsV2Stage,
    CatalogBuilderStage,
    CoverageOracleDebuggerStage,
    DbHackProductResolverStage,
    QrZoneDecodeStage,
    RowMaterializerStage,
    TrackToCatalogAssignmentStage,
    TrackletGraphMergeStage,
    V4FrameSamplingStage,
)


class PriceTagV4Pipeline(PriceTagCpuV1Pipeline):
    def __init__(self) -> None:
        self._config_path = Path(__file__).with_name("config.yaml")
        self._base_config = _load_yaml_config(self._config_path)
        pipeline_config = self._base_config.get("pipeline", {})
        self.name = str(pipeline_config.get("name", "price_tag_v4"))
        self.default_version = str(pipeline_config.get("version", "0.1.0"))
        self._stages = [
            FrameMetadataStage(),
            V4FrameSamplingStage(),
            YoloByteTrackStage(),
            FallbackHeuristicCandidateDetectionStage(),
            TrackletGraphMergeStage(),
            CropExtractionStage(),
            BarcodeQrDecodeStage(),
            QrZoneDecodeStage(),
            ZonalOcrStage(),
            TrackCropBankStage(),
            MultiViewFusionStage(),
            BarcodeBarsV2Stage(),
            FusedZonalDecodeStage(),
            CharConsensusStage(),
            DbHackProductResolverStage(),
            RowFusionStage(),
            CatalogBuilderStage(),
            TrackToCatalogAssignmentStage(),
            RowMaterializerStage(),
            CoverageOracleDebuggerStage(),
            CsvWriterStage(),
            PreviewWriterStage(),
            DebugManifestStage(),
        ]
