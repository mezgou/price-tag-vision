from __future__ import annotations

from pathlib import Path

from app.pipelines.price_tag_v4.pipeline import PriceTagV4Pipeline
from app.pipelines.price_tag_v4.stages.catalog_builder import (
    LayoutCatalogEntry,
    filename_keys,
    load_db_hack_catalog,
)
from app.pipelines.price_tag_v4.stages.row_materializer import (
    RowMaterializerConfig,
    RowMaterializerStage,
    materialize_assigned_row,
)
from app.pipelines.price_tag_v4.stages.track_to_catalog_assignment import (
    AssignmentConfig,
    TrackEvidence,
    solve_global_assignment,
)
from app.pipelines.registry import get_pipeline_registry
from app.schemas.detections import BoundingBox
from shared.csv_schema import CSV_COLUMNS


def test_price_tag_v4_pipeline_config_loads() -> None:
    pipeline = PriceTagV4Pipeline()

    assert pipeline.name == "price_tag_v4"
    assert pipeline.default_version == "0.1.0"
    assert pipeline._base_config["camera"]["undistort"] is True
    assert pipeline._base_config["catalog_builder_v4"]["db_hack_path"] == "data/db_hack.csv"
    assert "TrackToCatalogAssignmentStage" in [stage.name for stage in pipeline._stages]
    assert "RowMaterializerStage" in [stage.name for stage in pipeline._stages]


def test_registry_exposes_v4_without_removing_v2_v3() -> None:
    names = get_pipeline_registry().names()

    assert "price_tag_v2" in names
    assert "price_tag_v3" in names
    assert "price_tag_v4" in names


def test_v4_global_assignment_is_one_to_one() -> None:
    config = AssignmentConfig(
        enabled=True,
        gt_root=Path("unused"),
        catalog_spatial_mode=True,
        catalog_semantic_mode=False,
        same_video_only_in_spatial_mode=True,
        allow_spatial_split_assignments=True,
        spatial_split_min_iou=0.50,
        min_score=70.0,
        min_margin=2.0,
        min_spatial_iou=0.18,
        strong_spatial_iou=0.50,
    )
    entry_a = _entry(0, "1111111111116", BoundingBox(x_min=10, y_min=10, x_max=110, y_max=80))
    entry_b = _entry(1, "2222222222224", BoundingBox(x_min=210, y_min=10, x_max=310, y_max=80))
    tracks = [
        _track("track_a", BoundingBox(x_min=12, y_min=12, x_max=112, y_max=82)),
        _track("track_b", BoundingBox(x_min=212, y_min=12, x_max=312, y_max=82)),
        _track("track_dup", BoundingBox(x_min=14, y_min=14, x_max=114, y_max=84)),
    ]

    assignments = solve_global_assignment(tracks=tracks, entries=[entry_a, entry_b], config=config)

    assert len(assignments) == 2
    assert {item.entry.index for item in assignments} == {0, 1}
    assert len({item.track_id for item in assignments}) == 2


def test_v4_materializer_copies_catalog_only_after_assignment(pipeline_context) -> None:
    pipeline_context.config["row_materializer_v4"] = {"enabled": True, "output_unassigned": False}
    pipeline_context.csv_rows = []
    pipeline_context.artifacts["v4_assignments"] = []

    outcome = RowMaterializerStage().run(pipeline_context)

    assert outcome.output_summary["rows_count"] == 0
    assert pipeline_context.csv_rows == []


def test_v4_materializer_backfills_assigned_catalog_row(pipeline_context) -> None:
    catalog_row = {column: "" for column in CSV_COLUMNS}
    catalog_row.update(
        {
            "product_name": "Catalog Product",
            "price_card": "1199.99",
            "barcode": "4607124143901",
            "qr_code_barcode": "4607124143901",
        }
    )
    assignment = {
        "track_id": "merged_track_00001",
        "reasons": ["strong_spatial_iou"],
        "anchor_timestamp_ms": 1234,
        "anchor_bbox": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 90},
    }

    row, conflict = materialize_assigned_row(
        context=pipeline_context,
        assignment=assignment,
        catalog_row=catalog_row,
        track_fields={},
        config=RowMaterializerConfig(
            enabled=True,
            output_unassigned=False,
            copy_empty_catalog_values=False,
            default_absent_value="нет",
        ),
    )

    assert conflict is False
    assert row["barcode"] == "4607124143901"
    assert row["qr_code_barcode"] == "4607124143901"
    assert row["product_name"] == "Catalog Product"
    assert row["frame_timestamp"] == "1234"
    assert row["x_min"] == "10.0"


def test_v4_db_hack_loader_decodes_cp1251(tmp_path: Path) -> None:
    db_path = tmp_path / "db_hack.csv"
    db_path.write_bytes("fullname;code\r\nТовар тестовый;4607124143901\r\n".encode("cp1251"))

    catalog = load_db_hack_catalog(db_path)

    assert catalog.name_for("4607124143901") == "Товар тестовый"


def _entry(index: int, barcode: str, bbox: BoundingBox) -> LayoutCatalogEntry:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "filename": "video.mp4",
            "barcode": barcode,
            "qr_code_barcode": barcode,
            "x_min": str(bbox.x_min),
            "y_min": str(bbox.y_min),
            "x_max": str(bbox.x_max),
            "y_max": str(bbox.y_max),
        }
    )
    return LayoutCatalogEntry(
        index=index,
        row=row,
        filename_keys=filename_keys("video.mp4"),
        bbox=bbox,
        shelf_band=0,
        order_key=(0, float(bbox.x_min), float(bbox.y_min)),
    )


def _track(track_id: str, bbox: BoundingBox) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id,
        detections=(),
        fields={},
        raw_bboxes=((f"{track_id}_det", 1000, bbox),),
        shelf_band=0,
        order_x=float(bbox.x_min),
    )
