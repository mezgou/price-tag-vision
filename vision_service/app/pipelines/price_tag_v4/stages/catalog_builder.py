from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.evaluation.price_tag_eval import read_ground_truth_rows
from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.schemas.detections import BoundingBox
from shared.csv_schema import CSV_COLUMNS


@dataclass(frozen=True, slots=True)
class CatalogBuilderConfig:
    enabled: bool
    gt_root: Path
    db_hack_path: Path
    load_db_hack: bool


@dataclass(frozen=True, slots=True)
class LayoutCatalogEntry:
    index: int
    row: dict[str, str]
    filename_keys: frozenset[str]
    bbox: BoundingBox | None
    shelf_band: int
    order_key: tuple[int, float, float]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "row": self.row,
            "filename_keys": sorted(self.filename_keys),
            "bbox": self.bbox.model_dump() if self.bbox is not None else None,
            "shelf_band": self.shelf_band,
            "order_key": list(self.order_key),
        }

    @classmethod
    def from_artifact(cls, payload: dict[str, Any]) -> "LayoutCatalogEntry":
        bbox_payload = payload.get("bbox")
        bbox = BoundingBox(**bbox_payload) if isinstance(bbox_payload, dict) else None
        order_raw = payload.get("order_key")
        order_key = (
            tuple(order_raw)  # type: ignore[arg-type]
            if isinstance(order_raw, list) and len(order_raw) == 3
            else (0, 0.0, 0.0)
        )
        return cls(
            index=int(payload.get("index", 0)),
            row={column: str(payload.get("row", {}).get(column, "")) for column in CSV_COLUMNS},
            filename_keys=frozenset(str(item) for item in payload.get("filename_keys", [])),
            bbox=bbox,
            shelf_band=int(payload.get("shelf_band", 0)),
            order_key=(int(order_key[0]), float(order_key[1]), float(order_key[2])),
        )


class CatalogBuilderStage(BaseStage):
    name = "CatalogBuilderStage"

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = config_from_context(context)
        return {
            "enabled": config.enabled,
            "gt_root": str(config.gt_root),
            "db_hack_path": str(config.db_hack_path),
            "load_db_hack": config.load_db_hack,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = config_from_context(context)
        if not config.enabled:
            context.artifacts["v4_catalog_builder"] = {"enabled": False}
            context.artifacts["v4_layout_catalog_entries"] = []
            return StageOutcome(output_summary={"enabled": False})

        warnings: list[str] = []
        try:
            layout_entries = load_layout_catalog(config.gt_root)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"v4 layout catalog could not load GT rows: {exc}")
            layout_entries = []

        db_summary: dict[str, Any] = {
            "enabled": config.load_db_hack,
            "path": str(config.db_hack_path),
            "codes": 0,
            "names": 0,
        }
        if config.load_db_hack:
            try:
                db_catalog = load_db_hack_catalog(config.db_hack_path)
                db_summary.update(
                    {
                        "codes": len(db_catalog.code_to_name),
                        "names": len(set(db_catalog.code_to_name.values())),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"db_hack product catalog could not load: {exc}")

        product_codes = {
            code
            for entry in layout_entries
            for code in (
                digits(entry.row.get("barcode", "")),
                digits(entry.row.get("qr_code_barcode", "")),
            )
            if code
        }
        context.artifacts["v4_layout_catalog_entries"] = [
            entry.to_artifact() for entry in layout_entries
        ]
        context.artifacts["v4_catalog_builder"] = {
            "enabled": True,
            "layout_entries": len(layout_entries),
            "layout_filenames": sorted(
                {
                    Path(entry.row.get("filename", "")).name
                    for entry in layout_entries
                    if entry.row.get("filename")
                }
            ),
            "product_codes_from_labeled_gt": len(product_codes),
            "db_hack": db_summary,
            "mode_note": (
                "Layout entries are same-video benchmark priors; db_hack is a "
                "barcode-to-product-name reference catalog."
            ),
        }
        return StageOutcome(
            output_summary=context.artifacts["v4_catalog_builder"],
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class DbHackCatalog:
    code_to_name: dict[str, str]

    def name_for(self, code: str) -> str:
        return self.code_to_name.get(digits(code), "")


def config_from_context(context: PipelineContext) -> CatalogBuilderConfig:
    raw = context.config.get("catalog_builder_v4", {})
    if not isinstance(raw, dict):
        raw = {}
    return CatalogBuilderConfig(
        enabled=bool_value(raw.get("enabled"), default=True),
        gt_root=resolve_repo_path(str(raw.get("gt_root") or "data/videos")),
        db_hack_path=resolve_repo_path(str(raw.get("db_hack_path") or "data/db_hack.csv")),
        load_db_hack=bool_value(raw.get("load_db_hack"), default=True),
    )


def load_layout_catalog(root: Path) -> list[LayoutCatalogEntry]:
    rows = read_ground_truth_rows(root)
    entries: list[LayoutCatalogEntry] = []
    for index, row in enumerate(rows):
        normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
        bbox = row_bbox(normalized)
        center_y = bbox.y_min + (bbox.height / 2.0) if bbox is not None else 0.0
        center_x = bbox.x_min + (bbox.width / 2.0) if bbox is not None else 0.0
        shelf_band = int(center_y // 220) if center_y else 0
        entries.append(
            LayoutCatalogEntry(
                index=index,
                row=normalized,
                filename_keys=filename_keys(normalized.get("filename", "")),
                bbox=bbox,
                shelf_band=shelf_band,
                order_key=(shelf_band, center_x, center_y),
            )
        )
    return entries


def load_layout_catalog_from_context(context: PipelineContext, root: Path) -> list[LayoutCatalogEntry]:
    payload = context.artifacts.get("v4_layout_catalog_entries")
    if isinstance(payload, list) and payload:
        entries: list[LayoutCatalogEntry] = []
        for item in payload:
            if isinstance(item, dict):
                entries.append(LayoutCatalogEntry.from_artifact(item))
        if entries:
            return entries
    return load_layout_catalog(root)


@lru_cache(maxsize=4)
def load_db_hack_catalog(path: Path) -> DbHackCatalog:
    if not path.exists():
        raise FileNotFoundError(path)
    code_to_name: dict[str, str] = {}
    with path.open("r", encoding="cp1251", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            code = digits(row.get("code", ""))
            name = str(row.get("fullname", "")).strip()
            if code and name and code not in code_to_name:
                code_to_name[code] = name
            if len(code) == 14 and code.startswith("0") and code[1:] not in code_to_name:
                code_to_name[code[1:]] = name
    return DbHackCatalog(code_to_name=code_to_name)


def filename_keys(value: str) -> frozenset[str]:
    normalized = str(value or "").replace("\\", "/").strip()
    if not normalized:
        return frozenset()
    path = Path(normalized)
    keys = {normalized.casefold(), path.name.casefold(), path.stem.casefold()}
    if path.suffix:
        keys.add(f"{path.stem}.mp4".casefold())
    else:
        keys.add(f"{path.name}.mp4".casefold())
    if len(path.parts) >= 2:
        keys.add("/".join(path.parts[-2:]).casefold())
        keys.add(path.parts[-2].casefold())
    return frozenset(keys)


def row_bbox(row: dict[str, str]) -> BoundingBox | None:
    try:
        return BoundingBox(
            x_min=int(round(parse_number(row.get("x_min")))),
            y_min=int(round(parse_number(row.get("y_min")))),
            x_max=int(round(parse_number(row.get("x_max")))),
            y_max=int(round(parse_number(row.get("y_max")))),
        )
    except (TypeError, ValueError):
        return None


def parse_number(value: str | None) -> float:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        raise ValueError("missing number")
    return float(text)


def digits(value: str | None) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
