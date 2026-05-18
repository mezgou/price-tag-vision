from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

VISION_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_SERVICE_DIR.parent
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from app.pipelines.registry import get_pipeline_registry
from app.schemas.pipeline import ProcessRequest


class LocalDirectoryStorage:
    def __init__(
        self,
        *,
        input_key: str,
        input_video_path: Path,
        output_root: Path,
    ) -> None:
        self.input_key = input_key
        self.input_video_path = input_video_path
        self.output_root = output_root
        self.objects: dict[str, tuple[Path, str]] = {}

    def ensure_bucket(self) -> None:
        return None

    def ensure_object(self, key: str) -> None:
        if key != self.input_key or not self.input_video_path.exists():
            raise KeyError(key)

    def download_file(self, key: str, destination: Path) -> None:
        self.ensure_object(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.input_video_path, destination)

    def upload_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        destination = self.output_root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.objects[key] = (destination, content_type)


def main() -> int:
    args = _parse_args()
    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    job_id = args.job_id or _slugify(f"smoke-{video_path.stem}")
    input_key = f"inputs/{job_id}/{video_path.name}"
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    storage = LocalDirectoryStorage(
        input_key=input_key,
        input_video_path=video_path,
        output_root=output_root,
    )
    request = ProcessRequest(
        job_id=job_id,
        input_video_key=input_key,
        pipeline_name=args.pipeline,
        config=_build_request_config(args),
    )

    pipeline = get_pipeline_registry().resolve(args.pipeline, default_name=args.pipeline)
    response = pipeline.run(request, storage)

    preview_path = output_root / response.preview_key
    manifest_path = output_root / f"outputs/{job_id}/debug/pipeline_manifest.json"
    preview_payload = json.loads(preview_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contact_sheet_keys = preview_payload.get("debug_contact_sheet_keys", [])
    contact_sheet_path = (
        str(output_root / contact_sheet_keys[0]) if contact_sheet_keys else ""
    )

    summary = {
        "job_id": response.job_id,
        "video": str(video_path),
        "output_root": str(output_root),
        "csv_key": response.csv_key,
        "preview_key": response.preview_key,
        "manifest_key": f"outputs/{job_id}/debug/pipeline_manifest.json",
        "stats": response.stats,
        "debug_preview": {
            "debug_frame_keys": preview_payload.get("debug_frame_keys", []),
            "debug_overlay_keys": preview_payload.get("debug_overlay_keys", []),
            "debug_mask_keys": preview_payload.get("debug_mask_keys", []),
            "debug_crop_keys": preview_payload.get("debug_crop_keys", []),
            "debug_contact_sheet_keys": contact_sheet_keys,
            "crop_quality_summary": preview_payload.get("crop_quality_summary", {}),
        },
        "contact_sheet_path": contact_sheet_path,
        "manifest_stage_summaries": {
            stage["name"]: stage.get("output_summary", {})
            for stage in manifest_payload.get("stages", [])
            if stage.get("name")
            in {
                "HeuristicCandidateDetectionStage",
                "FallbackHeuristicCandidateDetectionStage",
                "YoloDetectionStage",
                "CropExtractionStage",
                "BarcodeQrDecodeStage",
                "RowFusionStage",
            }
        },
    }
    summary_path = output_root / f"outputs/{job_id}/smoke_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a price-tag pipeline directly on a local MP4 and write artifacts to a local directory.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to a local MP4 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/smoke"),
        help="Root directory where outputs/{job_id}/... will be written.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional explicit job id. Defaults to smoke-{video_stem}.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--pipeline", type=str, default="price_tag_cpu_v1")
    parser.add_argument("--max-candidates-per-frame", type=int, default=24)
    parser.add_argument("--max-total-crops", type=int, default=120)
    parser.add_argument("--max-crops-per-frame", type=int, default=16)
    return parser.parse_args()


def _build_request_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frame_sampling": {
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
        },
        "candidate_detection": {
            "max_candidates_per_frame": args.max_candidates_per_frame,
        },
        "crop_extraction": {
            "max_total_crops": args.max_total_crops,
            "max_crops_per_frame": args.max_crops_per_frame,
        },
    }


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "smoke-job"


if __name__ == "__main__":
    raise SystemExit(main())
