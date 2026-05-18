from __future__ import annotations
# ruff: noqa: E402

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

VISION_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_SERVICE_DIR.parent
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from app.yolo.price_tag_workflows import (
    build_evaluation_report,
    build_train_overrides,
    build_val_overrides,
    count_label_boxes,
    count_split_images,
    resolve_yolo_run_paths,
    write_json_report,
)


def main() -> int:
    args = _parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for this script. Run it with "
            "`uv run --with ultralytics python vision_service/scripts/train_yolo_price_tag_detector.py ...` "
            "or install ultralytics into a local dev environment."
        ) from error

    device_used, device_notes = _resolve_effective_device(
        requested_device=args.device,
        torch_module=torch,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )

    dataset_path = args.data.resolve()
    dataset_root = dataset_path.parent
    project_dir = args.project.resolve()
    run_paths = resolve_yolo_run_paths(project_dir=project_dir, run_name=args.name)
    run_paths.qa_dir.mkdir(parents=True, exist_ok=True)
    run_paths.export_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = build_train_overrides(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_used,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        plots=args.plots,
        workers=args.workers,
    )

    training_started_at = time.monotonic()
    model = YOLO(args.model)
    model.train(**train_kwargs)
    training_duration_sec = time.monotonic() - training_started_at

    best_path = run_paths.weights_dir / "best.pt"
    last_path = run_paths.weights_dir / "last.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt was not produced: {best_path}")

    best_model = YOLO(str(best_path))
    val_kwargs = build_val_overrides(
        data=str(dataset_path),
        imgsz=args.imgsz,
        conf=args.val_conf,
        iou=args.val_iou,
        project=str(run_paths.run_dir),
        name="val_eval",
        device=device_used,
        plots=True,
        workers=args.workers,
    )
    metrics = best_model.val(**val_kwargs)
    metric_values = _extract_metric_values(metrics)

    export_status = _export_model_artifacts(
        model=best_model,
        best_path=best_path,
        export_dir=run_paths.export_dir,
        imgsz=args.imgsz,
        export_openvino=not args.skip_openvino_export,
    )

    train_images = count_split_images(dataset_root / "images" / "train")
    val_images = count_split_images(dataset_root / "images" / "val")
    train_boxes = count_label_boxes(dataset_root / "labels" / "train")
    val_boxes = count_label_boxes(dataset_root / "labels" / "val")

    report = build_evaluation_report(
        model_path=best_path,
        dataset_path=dataset_path,
        run_dir=run_paths.run_dir,
        train_images=train_images,
        val_images=val_images,
        train_boxes=train_boxes,
        val_boxes=val_boxes,
        precision=metric_values.get("precision"),
        recall=metric_values.get("recall"),
        map50=metric_values.get("map50"),
        map50_95=metric_values.get("map50_95"),
        best_conf_suggestion=_suggest_best_conf(metric_values),
        notes=device_notes
        + [
            "This stage trains and evaluates the detector only. Runtime integration remains out of scope.",
            "Given the tiny dataset, qualitative QA is required alongside the reported metrics.",
        ],
        device_requested=args.device,
        device_used=device_used,
        training_duration_sec=training_duration_sec,
        val_output_dir=run_paths.val_eval_dir,
        results_csv_path=run_paths.run_dir / "results.csv",
        results_png_path=run_paths.run_dir / "results.png",
        export_status=export_status,
    )
    report["checkpoint_paths"] = {
        "best_pt": str(best_path),
        "last_pt": str(last_path),
    }
    write_json_report(run_paths.evaluation_report_path, report)

    print(report["checkpoint_paths"]["best_pt"])
    print(run_paths.evaluation_report_path)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a local YOLO11n price_tag detector without changing the runtime pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/datasets/price_tag_yolo_v1/dataset.yaml"),
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--name", type=str, default="price_tag_yolo11n_v1")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--val-conf", type=float, default=0.1)
    parser.add_argument("--val-iou", type=float, default=0.5)
    parser.add_argument("--skip-openvino-export", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser.parse_args()


def _resolve_effective_device(
    *,
    requested_device: str,
    torch_module: Any,
    allow_cpu_fallback: bool,
) -> tuple[str, list[str]]:
    stripped = requested_device.strip().lower()
    if stripped == "cpu":
        return "cpu", ["Training is explicitly pinned to CPU."]

    if torch_module.cuda.is_available():
        return requested_device, [
            f"CUDA is available. Training will use device '{requested_device}'."
        ]

    if allow_cpu_fallback:
        return "cpu", [
            "CUDA is not available in the current training environment.",
            f"Requested device '{requested_device}' was downgraded to CPU fallback.",
        ]

    raise RuntimeError(
        "CUDA is not available in the current training environment and "
        "--allow-cpu-fallback was not provided."
    )


def _extract_metric_values(metrics: Any) -> dict[str, float | None]:
    box_metrics = getattr(metrics, "box", None)
    return {
        "precision": _safe_metric(box_metrics, "mp", "p"),
        "recall": _safe_metric(box_metrics, "mr", "r"),
        "map50": _safe_metric(box_metrics, "map50"),
        "map50_95": _safe_metric(box_metrics, "map"),
    }


def _safe_metric(obj: Any, *attribute_names: str) -> float | None:
    for attribute_name in attribute_names:
        if obj is None or not hasattr(obj, attribute_name):
            continue
        value = getattr(obj, attribute_name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _suggest_best_conf(metric_values: dict[str, float | None]) -> float:
    recall = metric_values.get("recall")
    precision = metric_values.get("precision")
    if recall is not None and precision is not None and recall < precision:
        return 0.05
    return 0.10


def _export_model_artifacts(
    *,
    model: Any,
    best_path: Path,
    export_dir: Path,
    imgsz: int,
    export_openvino: bool,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "onnx": {"success": False, "path": None, "error": None},
        "openvino": {"success": False, "path": None, "error": None},
    }

    try:
        exported_onnx = Path(
            str(
                model.export(
                    format="onnx",
                    imgsz=imgsz,
                    dynamic=False,
                    simplify=True,
                )
            )
        )
        onnx_destination = export_dir / "best.onnx"
        shutil.copy2(exported_onnx, onnx_destination)
        status["onnx"] = {"success": True, "path": str(onnx_destination), "error": None}
    except Exception as error:  # noqa: BLE001
        status["onnx"] = {"success": False, "path": None, "error": str(error)}

    if not export_openvino:
        status["openvino"]["error"] = "OpenVINO export was skipped by CLI flag."
        return status

    try:
        exported_openvino = Path(
            str(
                model.export(
                    format="openvino",
                    imgsz=imgsz,
                )
            )
        )
        openvino_destination = export_dir / "best_openvino_model"
        if openvino_destination.exists():
            shutil.rmtree(openvino_destination)
        if exported_openvino.is_dir():
            shutil.copytree(exported_openvino, openvino_destination)
        else:
            openvino_destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exported_openvino, openvino_destination / exported_openvino.name)
        status["openvino"] = {
            "success": True,
            "path": str(openvino_destination),
            "error": None,
        }
    except Exception as error:  # noqa: BLE001
        status["openvino"] = {"success": False, "path": None, "error": str(error)}

    return status


if __name__ == "__main__":
    raise SystemExit(main())
