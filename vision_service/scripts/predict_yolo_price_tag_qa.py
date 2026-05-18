from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VISION_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_SERVICE_DIR.parent
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from app.yolo.price_tag_workflows import (
    ContactSheetItem,
    add_overlay_text,
    discover_unlabeled_videos,
    format_conf_tag,
    infer_run_dir_from_model_path,
    resolve_yolo_run_paths,
    sample_video_frames,
    save_contact_sheet,
    save_jpeg,
    summarize_detection_counts,
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
            "`uv run --with ultralytics python vision_service/scripts/predict_yolo_price_tag_qa.py ...` "
            "or install ultralytics into a local dev environment."
        ) from error

    device_used = _resolve_prediction_device(requested_device=args.device, torch_module=torch)

    model_path = args.model.resolve()
    dataset_path = args.data.resolve()
    dataset_root = dataset_path.parent
    run_dir = infer_run_dir_from_model_path(model_path)
    run_paths = resolve_yolo_run_paths(project_dir=run_dir.parent, run_name=run_dir.name)
    run_paths.val_prediction_overlays_dir.mkdir(parents=True, exist_ok=True)
    run_paths.real_video_overlays_dir.mkdir(parents=True, exist_ok=True)
    run_paths.unlabeled_overlays_dir.mkdir(parents=True, exist_ok=True)
    run_paths.qa_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    qa_artifacts: dict[str, Any] = {}
    qa_summary: dict[str, Any] = {}

    val_items, val_summary = _predict_val_images(
        model=model,
        dataset_root=dataset_root,
        overlays_dir=run_paths.val_prediction_overlays_dir,
        imgsz=args.imgsz,
        conf=args.val_conf,
        iou=args.iou,
        device=device_used,
    )
    val_contact_sheet_path = run_paths.qa_dir / "val_prediction_contact_sheet.jpg"
    save_contact_sheet(
        path=val_contact_sheet_path,
        items=val_items,
        thumb_width=280,
        thumb_height=420,
    )
    qa_artifacts["val_prediction_overlays_dir"] = str(run_paths.val_prediction_overlays_dir)
    qa_artifacts["val_prediction_contact_sheet"] = str(val_contact_sheet_path)
    qa_summary["val_predictions"] = val_summary

    real_video_path = args.real_video.resolve()
    real_frames = sample_video_frames(
        video_path=real_video_path,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        orientation_mode=args.orientation_mode,
    )
    real_video_artifacts, real_video_summary = _predict_sampled_frames_by_conf(
        model=model,
        frames=real_frames,
        output_root=run_paths.real_video_overlays_dir,
        imgsz=args.imgsz,
        iou=args.iou,
        device=device_used,
        confidences=args.real_video_confidences,
        contact_sheet_prefix="real_video_prediction_contact_sheet",
        qa_dir=run_paths.qa_dir,
    )
    qa_artifacts.update(real_video_artifacts)
    qa_summary["real_video_predictions"] = real_video_summary

    unlabeled_videos = discover_unlabeled_videos(args.unlabeled_root.resolve())
    if unlabeled_videos:
        unlabeled_items, unlabeled_summary = _predict_unlabeled_videos(
            model=model,
            video_paths=unlabeled_videos,
            output_root=run_paths.unlabeled_overlays_dir,
            qa_dir=run_paths.qa_dir,
            imgsz=args.imgsz,
            iou=args.iou,
            conf=args.unlabeled_conf,
            device=device_used,
            sample_fps=args.sample_fps,
            max_frames_per_video=max(1, args.max_frames // max(len(unlabeled_videos), 1)),
            orientation_mode=args.orientation_mode,
        )
        unlabeled_contact_sheet_path = run_paths.qa_dir / "unlabeled_prediction_contact_sheet.jpg"
        save_contact_sheet(
            path=unlabeled_contact_sheet_path,
            items=unlabeled_items,
            thumb_width=280,
            thumb_height=420,
        )
        qa_artifacts["unlabeled_prediction_overlays_dir"] = str(run_paths.unlabeled_overlays_dir)
        qa_artifacts["unlabeled_prediction_contact_sheet"] = str(unlabeled_contact_sheet_path)
        qa_summary["unlabeled_predictions"] = unlabeled_summary

    report = _load_existing_report(run_paths.evaluation_report_path)
    report["qa_artifacts"] = {**report.get("qa_artifacts", {}), **qa_artifacts}
    report["qa_summary"] = qa_summary
    report["device_used_for_prediction"] = device_used
    write_json_report(run_paths.evaluation_report_path, report)
    print(json.dumps(qa_artifacts, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative QA artifacts for a trained local YOLO price_tag detector."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/price_tag_yolo11n_v1/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/datasets/price_tag_yolo_v1/dataset.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--val-conf", type=float, default=0.1)
    parser.add_argument("--unlabeled-conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--real-video", type=Path, default=Path("data/videos/43_15/43_15.mp4"))
    parser.add_argument("--unlabeled-root", type=Path, default=Path("data/videos/unlabeled"))
    parser.add_argument("--orientation-mode", type=str, default="rotate_90_ccw")
    parser.add_argument(
        "--real-video-confidences",
        nargs="*",
        type=float,
        default=[0.05, 0.1, 0.25],
    )
    return parser.parse_args()


def _resolve_prediction_device(*, requested_device: str, torch_module: Any) -> str:
    stripped = requested_device.strip().lower()
    if stripped == "cpu":
        return "cpu"
    if torch_module.cuda.is_available():
        return requested_device
    return "cpu"


def _predict_val_images(
    *,
    model: Any,
    dataset_root: Path,
    overlays_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> tuple[list[ContactSheetItem], dict[str, Any]]:
    image_paths = sorted((dataset_root / "images" / "val").glob("*.jpg"))
    items: list[ContactSheetItem] = []
    detection_counts: list[int] = []

    for image_path in image_paths:
        prediction = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            save=False,
        )[0]
        detections_count = 0 if prediction.boxes is None else len(prediction.boxes)
        detection_counts.append(detections_count)
        overlay = add_overlay_text(
            prediction.plot(),
            title=f"{image_path.stem} conf={conf:.2f}",
            subtitle=f"detections={detections_count}",
        )
        overlay_path = overlays_dir / image_path.name
        save_jpeg(overlay_path, overlay)
        items.append(
            ContactSheetItem(
                title=image_path.stem,
                subtitle=f"det={detections_count} conf={conf:.2f}",
                image=overlay,
            )
        )

    return items, summarize_detection_counts(detection_counts)


def _predict_sampled_frames_by_conf(
    *,
    model: Any,
    frames: list[Any],
    output_root: Path,
    imgsz: int,
    iou: float,
    device: str,
    confidences: list[float],
    contact_sheet_prefix: str,
    qa_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    artifacts: dict[str, str] = {}
    summary: dict[str, Any] = {"frames_sampled": len(frames), "by_confidence": {}}

    for confidence in confidences:
        conf_tag = format_conf_tag(confidence)
        overlay_dir = output_root / f"conf_{conf_tag}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        items: list[ContactSheetItem] = []
        detection_counts: list[int] = []

        for frame in frames:
            prediction = model.predict(
                source=frame.frame,
                imgsz=imgsz,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
                save=False,
            )[0]
            detections_count = 0 if prediction.boxes is None else len(prediction.boxes)
            detection_counts.append(detections_count)
            overlay = add_overlay_text(
                prediction.plot(),
                title=f"{frame.source_path.stem} ts={frame.timestamp_ms}ms conf={confidence:.2f}",
                subtitle=f"detections={detections_count}",
            )
            overlay_path = overlay_dir / f"{frame.display_name}.jpg"
            save_jpeg(overlay_path, overlay)
            items.append(
                ContactSheetItem(
                    title=frame.display_name,
                    subtitle=f"det={detections_count} conf={confidence:.2f}",
                    image=overlay,
                )
            )

        contact_sheet_path = qa_dir / f"{contact_sheet_prefix}_conf_{conf_tag}.jpg"
        save_contact_sheet(
            path=contact_sheet_path,
            items=items,
            thumb_width=280,
            thumb_height=420,
        )
        artifacts[f"{contact_sheet_prefix}_conf_{conf_tag}"] = str(contact_sheet_path)
        summary["by_confidence"][f"{confidence:.2f}"] = {
            "contact_sheet": str(contact_sheet_path),
            "summary": summarize_detection_counts(detection_counts),
        }

    return artifacts, summary


def _predict_unlabeled_videos(
    *,
    model: Any,
    video_paths: list[Path],
    output_root: Path,
    qa_dir: Path,
    imgsz: int,
    iou: float,
    conf: float,
    device: str,
    sample_fps: float,
    max_frames_per_video: int,
    orientation_mode: str,
) -> tuple[list[ContactSheetItem], dict[str, Any]]:
    items: list[ContactSheetItem] = []
    summary: dict[str, Any] = {}

    for video_path in video_paths:
        sampled_frames = sample_video_frames(
            video_path=video_path,
            sample_fps=sample_fps,
            max_frames=max_frames_per_video,
            orientation_mode=orientation_mode,
        )
        detection_counts: list[int] = []
        video_overlay_dir = output_root / video_path.stem
        video_overlay_dir.mkdir(parents=True, exist_ok=True)

        for frame in sampled_frames:
            prediction = model.predict(
                source=frame.frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                save=False,
            )[0]
            detections_count = 0 if prediction.boxes is None else len(prediction.boxes)
            detection_counts.append(detections_count)
            overlay = add_overlay_text(
                prediction.plot(),
                title=f"{video_path.stem} ts={frame.timestamp_ms}ms conf={conf:.2f}",
                subtitle=f"detections={detections_count}",
            )
            overlay_path = video_overlay_dir / f"{frame.display_name}.jpg"
            save_jpeg(overlay_path, overlay)
            items.append(
                ContactSheetItem(
                    title=f"{video_path.stem} ts={frame.timestamp_ms}",
                    subtitle=f"det={detections_count} conf={conf:.2f}",
                    image=overlay,
                )
            )

        summary[video_path.name] = summarize_detection_counts(detection_counts)

    return items, summary


def _load_existing_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
