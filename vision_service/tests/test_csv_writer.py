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
