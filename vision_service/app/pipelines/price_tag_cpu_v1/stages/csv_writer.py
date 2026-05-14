from __future__ import annotations

import csv
import io
from typing import Any

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from shared.csv_schema import CSV_COLUMNS


class CsvWriterStage(BaseStage):
    name = "CsvWriterStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "rows_buffered": len(context.csv_rows),
            "columns_count": len(CSV_COLUMNS),
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in context.csv_rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

        csv_key = context.artifact_writer.upload_text(
            "result.csv",
            buffer.getvalue(),
            "text/csv; charset=utf-8",
        )
        context.artifacts["csv_key"] = csv_key

        return StageOutcome(
            output_summary={
                "csv_key": csv_key,
                "rows_written": len(context.csv_rows),
                "columns_count": len(CSV_COLUMNS),
            }
        )
