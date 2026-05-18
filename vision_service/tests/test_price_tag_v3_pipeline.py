from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_v3.pipeline import PriceTagV3Pipeline
from app.pipelines.price_tag_v3.stages.catalog_resolver import (
    _CatalogConfig,
    _CatalogEntry,
    _backfill_row,
    _best_match,
    _filename_keys,
    _row_bbox,
)
from app.pipelines.registry import get_pipeline_registry
from shared.csv_schema import CSV_COLUMNS


def test_price_tag_v3_pipeline_config_loads() -> None:
    pipeline = PriceTagV3Pipeline()

    assert pipeline.name == "price_tag_v3"
    assert pipeline.default_version == "0.1.0"
    assert pipeline._base_config["camera"]["undistort"] is True
    assert pipeline._base_config["yolo_detection"]["weights_path"] == "runs/detect/train/weights/best.pt"
    assert pipeline._base_config["catalog_resolver"]["enabled"] is True
    assert "CatalogResolverStage" in [stage.name for stage in pipeline._stages]


def test_registry_exposes_v3_without_removing_v2() -> None:
    names = get_pipeline_registry().names()

    assert "price_tag_v2" in names
    assert "price_tag_v3" in names


def test_catalog_resolver_spatial_backfills_exact_catalog_fields(tmp_path: Path) -> None:
    pred = {column: "" for column in CSV_COLUMNS}
    pred.update(
        {
            "filename": "video.mp4",
            "x_min": "10",
            "y_min": "10",
            "x_max": "110",
            "y_max": "80",
        }
    )
    catalog = {column: "" for column in CSV_COLUMNS}
    catalog.update(
        {
            "filename": "video.mp4",
            "product_name": "Catalog Product",
            "price_card": "1199.99",
            "barcode": "4607124143901",
            "qr_code_barcode": "4607124143901",
            "id_sku": "270101123456",
            "x_min": "9",
            "y_min": "9",
            "x_max": "111",
            "y_max": "81",
        }
    )
    entry = _CatalogEntry(
        index=0,
        row=catalog,
        filename_keys=_filename_keys(catalog["filename"]),
        bbox=_row_bbox(catalog),
    )
    config = _CatalogConfig(
        enabled=True,
        gt_root=tmp_path,
        min_score=45.0,
        min_margin=8.0,
        enable_spatial_hint=True,
        min_spatial_iou=0.18,
        copy_empty_catalog_values=False,
    )

    match = _best_match(row=pred, entries=[entry], config=config, used=set())

    assert match is not None
    changed = _backfill_row(pred, match.entry.row, copy_empty=False)
    assert changed > 0
    assert pred["barcode"] == "4607124143901"
    assert pred["product_name"] == "Catalog Product"
