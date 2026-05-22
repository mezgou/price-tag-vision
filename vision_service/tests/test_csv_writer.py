from __future__ import annotations

import csv
import io

from app.pipelines.price_tag_cpu_v1.stages.csv_writer import CsvWriterStage
from shared.csv_schema import CSV_COLUMNS


def test_csv_writer_uses_canonical_header(pipeline_context, in_memory_storage) -> None:
    stage = CsvWriterStage()

    outcome = stage.run(pipeline_context)

    csv_key = outcome.output_summary["csv_key"]
    payload, content_type = in_memory_storage.objects[csv_key]
    reader = csv.reader(io.StringIO(payload.decode("utf-8")))
    header = next(reader)

    assert content_type == "text/csv; charset=utf-8"
    assert header == CSV_COLUMNS


def test_csv_writer_ignores_catalog_guess_metadata(
    pipeline_context,
    in_memory_storage,
) -> None:
    pipeline_context.csv_rows = [
        {
            "filename": "video.mp4",
            "catalog_match_status": "catalog_guess",
            "catalog_guess_name": "Likely Wine",
        }
    ]

    outcome = CsvWriterStage().run(pipeline_context)

    payload, _content_type = in_memory_storage.objects[outcome.output_summary["csv_key"]]
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    row = next(reader)

    assert reader.fieldnames == CSV_COLUMNS
    assert "catalog_guess_name" not in row
    assert row["filename"] == "video.mp4"
