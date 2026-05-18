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

from app.evaluation.price_tag_eval import evaluate_prediction_csv


def main() -> int:
    args = _parse_args()
    report = evaluate_prediction_csv(
        pred_csv_path=args.pred.resolve(),
        gt_root=args.gt.resolve(),
        iou_threshold=args.iou,
        row_field_accuracy_threshold=args.field_accuracy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a price-tag prediction CSV against local GT CSV files."
    )
    parser.add_argument("--pred", type=Path, required=True, help="Prediction CSV path.")
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="GT CSV file or root directory containing official labeled CSV files.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--field-accuracy", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

