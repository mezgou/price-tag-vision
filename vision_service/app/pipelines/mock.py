from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
from typing import Any

from PIL import Image, ImageDraw

from shared.csv_schema import CSV_COLUMNS


@dataclass(slots=True)
class MockArtifacts:
    csv_bytes: bytes
    preview_bytes: bytes
    crop_bytes: bytes
    stats: dict[str, Any]


def build_mock_artifacts(
    job_id: str,
    input_video_key: str,
    pipeline_name: str,
    pipeline_version: str,
) -> MockArtifacts:
    row = {column: "нет" for column in CSV_COLUMNS}
    row.update(
        {
            "filename": input_video_key.rsplit("/", maxsplit=1)[-1],
            "product_name": "Mock price tag",
            "price_default": "199.99",
            "price_card": "179.99",
            "price_discount": "169.99",
            "discount_amount": "30.00",
            "barcode": "4600000000000",
            "frame_timestamp": "1200",
            "x_min": "84",
            "y_min": "128",
            "x_max": "420",
            "y_max": "286",
        }
    )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(row)

    preview_payload = {
        "job_id": job_id,
        "columns": CSV_COLUMNS,
        "rows": [row],
    }

    crop_image = Image.new("RGB", (640, 360), color="#f2eadf")
    draw = ImageDraw.Draw(crop_image)
    draw.rounded_rectangle((84, 128, 420, 286), radius=18, outline="#2b6a53", width=5)
    draw.text((110, 155), "mock pipeline preview", fill="#20170f")

    crop_buffer = io.BytesIO()
    crop_image.save(crop_buffer, format="JPEG", quality=92)

    stats = {
        "detected_price_tags": 1,
        "pipeline_name": pipeline_name,
        "pipeline_version": pipeline_version,
    }

    return MockArtifacts(
        csv_bytes=csv_buffer.getvalue().encode("utf-8"),
        preview_bytes=json.dumps(preview_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        crop_bytes=crop_buffer.getvalue(),
        stats=stats,
    )
