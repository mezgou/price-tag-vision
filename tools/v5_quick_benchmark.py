from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_SERVICE_DIR = REPO_ROOT / "vision_service"
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from app.evaluation.price_tag_eval import evaluate_prediction_csv  # noqa: E402

NO_VALUE = "нет"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    video: Path
    gt: Path
    sample_fps: float
    max_frames: int
    max_total_crops: int
    max_crops_per_frame: int


DEFAULT_CASES = {
    "43_15_micro": BenchmarkCase(
        name="43_15_micro",
        video=REPO_ROOT / "data/videos/43_15/43_15.mp4",
        gt=REPO_ROOT / "data/videos/43_15/43_15.csv",
        sample_fps=5.0,
        max_frames=5,
        max_total_crops=20,
        max_crops_per_frame=8,
    ),
    "26_12-20_wine": BenchmarkCase(
        name="26_12-20_wine",
        video=REPO_ROOT / "data/videos/26_12-20/26_12-20.mp4",
        gt=REPO_ROOT / "data/videos/26_12-20/26_12-20.csv",
        sample_fps=5.0,
        max_frames=60,
        max_total_crops=160,
        max_crops_per_frame=16,
    ),
    "26_12-20_quality": BenchmarkCase(
        name="26_12-20_quality",
        video=REPO_ROOT / "data/videos/26_12-20/26_12-20.mp4",
        gt=REPO_ROOT / "data/videos/26_12-20/26_12-20.csv",
        sample_fps=5.0,
        max_frames=100,
        max_total_crops=600,
        max_crops_per_frame=40,
    ),
    "26_12-20_quality900": BenchmarkCase(
        name="26_12-20_quality900",
        video=REPO_ROOT / "data/videos/26_12-20/26_12-20.mp4",
        gt=REPO_ROOT / "data/videos/26_12-20/26_12-20.csv",
        sample_fps=5.0,
        max_frames=100,
        max_total_crops=900,
        max_crops_per_frame=40,
    ),
    "26_12-20_full5fps": BenchmarkCase(
        name="26_12-20_full5fps",
        video=REPO_ROOT / "data/videos/26_12-20/26_12-20.mp4",
        gt=REPO_ROOT / "data/videos/26_12-20/26_12-20.csv",
        sample_fps=5.0,
        max_frames=500,
        max_total_crops=1800,
        max_crops_per_frame=48,
    ),
    "25_12-20_quick": BenchmarkCase(
        name="25_12-20_quick",
        video=REPO_ROOT / "data/videos/25_12-20/25_12-20.mp4",
        gt=REPO_ROOT / "data/videos/25_12-20/25_12-20.csv",
        sample_fps=3.0,
        max_frames=8,
        max_total_crops=32,
        max_crops_per_frame=8,
    ),
    "25_2-10_quick": BenchmarkCase(
        name="25_2-10_quick",
        video=REPO_ROOT / "data/videos/25_2-10/25_2-10.mp4",
        gt=REPO_ROOT / "data/videos/25_2-10/25_2-10.csv",
        sample_fps=3.0,
        max_frames=8,
        max_total_crops=32,
        max_crops_per_frame=8,
    ),
    "49_5_quick": BenchmarkCase(
        name="49_5_quick",
        video=REPO_ROOT / "data/videos/49_5/49_5.mp4",
        gt=REPO_ROOT / "data/videos/49_5/49_5.csv",
        sample_fps=3.0,
        max_frames=8,
        max_total_crops=32,
        max_crops_per_frame=8,
    ),
}

PRESETS = {
    "micro": ("43_15_micro",),
    "quick": ("43_15_micro", "26_12-20_wine"),
    "multi": (
        "25_12-20_quick",
        "25_2-10_quick",
        "26_12-20_wine",
        "43_15_micro",
        "49_5_quick",
    ),
}

STAGE_NAMES = (
    "BarcodeQrDecodeStage",
    "QrZoneDecodeStage",
    "V5BarcodeBarsCropDecodeStage",
    "V5DecodedSymbolGateStage",
    "V5BlockOcrCatalogStage",
    "V5RowConfidenceGateStage",
)


def main() -> int:
    args = _parse_args()
    case_names = _selected_case_names(args)
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    extra_config = _load_config(args.config_json, args.config_file)
    results = []
    for case_name in case_names:
        case = DEFAULT_CASES[case_name]
        results.append(
            run_case(
                case,
                pipeline=args.pipeline,
                output_root=output_root,
                run_id=args.run_id,
                extra_config=extra_config,
            )
        )

    summary = {
        "pipeline": args.pipeline,
        "run_id": args.run_id,
        "preset": args.preset,
        "cases": results,
    }
    summary_path = output_root / f"benchmark_{args.run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(_markdown_table(results))
    return 0


def run_case(
    case: BenchmarkCase,
    *,
    pipeline: str,
    output_root: Path,
    run_id: str,
    extra_config: dict[str, Any],
) -> dict[str, Any]:
    job_id = f"bench-{run_id}-{case.name}".replace("_", "-")
    config = _case_config(case)
    _deep_merge(config, extra_config)

    smoke_script = VISION_SERVICE_DIR / "scripts/smoke_price_tag_pipeline.py"
    command = [
        sys.executable,
        str(smoke_script),
        "--video",
        str(case.video),
        "--output-dir",
        str(output_root),
        "--job-id",
        job_id,
        "--pipeline",
        pipeline,
        "--sample-fps",
        str(case.sample_fps),
        "--max-frames",
        str(case.max_frames),
        "--max-total-crops",
        str(case.max_total_crops),
        "--max-crops-per-frame",
        str(case.max_crops_per_frame),
        "--config-json",
        json.dumps(config, ensure_ascii=False),
    ]
    completed = subprocess.run(
        command,
        # The pipeline config stores model/catalog paths relative to repo root
        # (for example runs/detect/train/weights/best.pt).
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    csv_path = output_root / "outputs" / job_id / "result.csv"
    manifest_path = output_root / "outputs" / job_id / "debug/pipeline_manifest.json"
    rows = _read_rows(csv_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_summaries = _stage_summaries(manifest)
    stats = manifest.get("stats", {})
    eval_report = (
        evaluate_prediction_csv(pred_csv_path=csv_path, gt_root=case.gt)
        if case.gt.exists()
        else {}
    )

    return {
        "case": case.name,
        "job_id": job_id,
        "csv": str(csv_path),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "barcode_non_empty": _non_empty_count(rows, "barcode"),
        "qr_code_barcode_non_empty": _non_empty_count(rows, "qr_code_barcode"),
        "id_sku_non_empty": _non_empty_count(rows, "id_sku"),
        "product_name_non_empty": _non_empty_count(rows, "product_name"),
        "decoded_symbols_total": stats.get("decoded_symbols_total", 0),
        "decoded_qr_total": stats.get("decoded_qr_total", 0),
        "decoded_barcode_total": stats.get("decoded_barcode_total", 0),
        "runtime_ms": stats.get("duration_ms", 0),
        "stage_summaries": stage_summaries,
        "eval": {
            key: eval_report.get(key)
            for key in (
                "score",
                "mean_field_accuracy",
                "matched_rows",
                "correct_rows",
                "mean_iou",
                "pred_rows",
                "gt_rows",
            )
            if key in eval_report
        },
        "smoke_stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
    }


def _case_config(case: BenchmarkCase) -> dict[str, Any]:
    max_crops = case.max_total_crops
    return {
        "barcode_qr_decode": {"max_crops": max_crops},
        "barcode_qr_zone_decode": {"max_crops": max_crops},
        "v5_barcode_bars_decode": {"max_crops": max_crops},
        "v5_block_ocr": {"max_crops": max_crops},
        "top_k_crop_selection": {"max_total_crops": max_crops},
        "crop_extraction": {
            "debug_save_crops": False,
            "debug_save_contact_sheet": False,
        },
    }


def _stage_summaries(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage in manifest.get("stages", []):
        name = stage.get("name")
        if name not in STAGE_NAMES:
            continue
        out[name] = {
            "duration_ms": stage.get("duration_ms", 0),
            **(stage.get("output_summary", {}) or {}),
        }
    return out


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _non_empty_count(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if _has_value(row.get(field, "")))


def _has_value(value: str | None) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.casefold() not in {NO_VALUE, "none", "n/a", "-"}


def _markdown_table(results: list[dict[str, Any]]) -> str:
    headers = [
        "case",
        "rows",
        "barcode",
        "qr",
        "sku",
        "name",
        "dec",
        "qr_dec",
        "bc_dec",
        "score",
        "field_acc",
        "matched",
        "correct",
        "iou",
        "runtime_s",
    ]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for result in results:
        ev = result.get("eval", {})
        lines.append(
            "|"
            + "|".join(
                [
                    str(result["case"]),
                    str(result["rows"]),
                    str(result["barcode_non_empty"]),
                    str(result["qr_code_barcode_non_empty"]),
                    str(result["id_sku_non_empty"]),
                    str(result["product_name_non_empty"]),
                    str(result["decoded_symbols_total"]),
                    str(result["decoded_qr_total"]),
                    str(result["decoded_barcode_total"]),
                    _fmt(ev.get("score")),
                    _fmt(ev.get("mean_field_accuracy")),
                    str(ev.get("matched_rows", "")),
                    str(ev.get("correct_rows", "")),
                    _fmt(ev.get("mean_iou")),
                    _fmt(float(result.get("runtime_ms", 0)) / 1000.0),
                ]
            )
            + "|"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _selected_case_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.cases:
        names = tuple(args.cases)
    else:
        names = PRESETS[args.preset]
    unknown = [name for name in names if name not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"Unknown benchmark case(s): {', '.join(unknown)}")
    return names


def _load_config(config_json: str, config_file: Path | None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if config_json.strip():
        loaded = json.loads(config_json)
        if not isinstance(loaded, dict):
            raise ValueError("--config-json must decode to an object")
        _deep_merge(config, loaded)
    if config_file is not None:
        loaded = json.loads(config_file.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise ValueError("--config-file must contain an object")
        _deep_merge(config, loaded)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge(current, value)
        else:
            base[key] = value
    return base


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast v5 benchmark matrix.")
    parser.add_argument("--pipeline", default="price_tag_v5")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--cases", nargs="*", default=())
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--output-root", default="artifacts/v5_quick_benchmark")
    parser.add_argument("--config-json", default="")
    parser.add_argument("--config-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
