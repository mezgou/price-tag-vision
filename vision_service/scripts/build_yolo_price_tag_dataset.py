from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

VISION_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_SERVICE_DIR.parent
if str(VISION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_SERVICE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from app.datasets.yolo_price_tag_dataset import build_yolo_price_tag_dataset
from app.pipelines.price_tag_cpu_v1.orientation import SUPPORTED_ORIENTATION_MODES


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    dataset_path_value = (
        args.output_dir.as_posix()
        if not args.output_dir.is_absolute()
        else str(output_dir)
    )

    report = build_yolo_price_tag_dataset(
        input_root=args.input_root.resolve(),
        output_dir=output_dir,
        orientation_mode=args.orientation_mode,
        train_videos=args.train_videos,
        val_videos=args.val_videos,
        make_orientation_qa=args.make_orientation_qa,
        orientation_review_note=args.orientation_review_note,
        dataset_path_value=dataset_path_value,
        propagate_frames=args.propagate_frames,
        propagation_ncc_threshold=args.propagation_ncc_threshold,
        propagation_frame_stride=args.propagation_frame_stride,
        keyframe_holdout_stride=args.keyframe_holdout_stride,
        undistort=args.undistort,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local YOLO-format price_tag dataset from official labeled video/CSV pairs."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/videos"),
        help="Root directory that contains labeled video folders with MP4 + CSV pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/price_tag_yolo_v1"),
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--orientation-mode",
        type=str,
        default="none",
        choices=SUPPORTED_ORIENTATION_MODES,
        help=(
            "Orientation mode applied to frames and transformed GT boxes. "
            "Use 'none' for the official raw 3840x2160 CSV coordinate space."
        ),
    )
    parser.add_argument(
        "--make-orientation-qa",
        action="store_true",
        help="Write comparison overlays under qa/orientation_check for each supported mode.",
    )
    parser.add_argument(
        "--train-videos",
        nargs="*",
        default=None,
        help="Optional explicit train split as video stems.",
    )
    parser.add_argument(
        "--val-videos",
        nargs="*",
        default=None,
        help="Optional explicit val split as video stems.",
    )
    parser.add_argument(
        "--orientation-review-note",
        type=str,
        default=None,
        help=(
            "Optional note written into qa/dataset_report.json after visual QA, "
            "for example: 'rotate_90_ccw looks correct on sampled overlays'."
        ),
    )
    parser.add_argument(
        "--propagate-frames",
        type=int,
        default=20,
        help=(
            "Template-match each CSV bbox onto +/-N neighboring frames. "
            "Use 0 to keep only the official keyframes."
        ),
    )
    parser.add_argument(
        "--propagation-ncc-threshold",
        type=float,
        default=0.42,
        help="Minimum cv2.TM_CCOEFF_NORMED score for propagated labels.",
    )
    parser.add_argument(
        "--propagation-frame-stride",
        type=int,
        default=1,
        help=(
            "Only keep every Nth neighboring frame during propagation. "
            "Use values > 1 to reduce near-duplicate train images."
        ),
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help=(
            "Apply lens-distortion correction (camera model) before "
            "orientation. Produces an undistorted + upright dataset; "
            "GT boxes are mapped through the same chain. Recommended."
        ),
    )
    parser.add_argument(
        "--keyframe-holdout-stride",
        type=int,
        default=0,
        help=(
            "Move every Nth keyframe group from train videos into val to create a "
            "more realistic validation set without training on adjacent propagated frames."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
