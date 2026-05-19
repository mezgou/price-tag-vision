"""v5 MVP quality report — one command, all the honest numbers.

Beyond the contest score (physically capped on this footage), an MVP is
judged on: how many real tags it surfaces (recall), how trustworthy each
emitted identity is (name precision — the hard project invariant), and how
well confidence triages the output. This prints all of that as JSON.

Usage:
  uv run --project vision_service python tools/v5_mvp_report.py \
    --pred  artifacts/<run>/outputs/<job>/result.csv \
    --gt    data/videos/26_12-20/26_12-20.csv \
    [--side-car artifacts/<run>/outputs/<job>/debug/row_confidence.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[1] / "vision_service"
if str(VISION) not in sys.path:
    sys.path.insert(0, str(VISION))
if str(VISION.parent) not in sys.path:
    sys.path.insert(1, str(VISION.parent))

from app.evaluation.price_tag_eval import (  # noqa: E402
    _row_bbox,
    _similarity,
    evaluate_prediction_csv,
)
from app.utils.image_processing import bbox_iou  # noqa: E402


def _norm(v: str) -> str:
    return " ".join(str(v or "").split()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--side-car", type=Path, default=None)
    args = ap.parse_args()

    rep = evaluate_prediction_csv(
        pred_csv_path=args.pred.resolve(), gt_root=args.gt.resolve()
    )
    pred = list(csv.DictReader(args.pred.open(encoding="utf-8-sig")))
    gt = list(csv.DictReader(args.gt.open(encoding="utf-8-sig")))

    p_boxes = [b for r in pred if (b := _row_bbox(r))]
    geo_recall = sum(
        1
        for g in gt
        if (gb := _row_bbox(g))
        and any(bbox_iou(gb, pb) >= 0.5 for pb in p_boxes)
    )

    # Name precision: among matched rows where BOTH pred and GT carry a
    # product_name, fraction the eval would accept (>=0.85). This is the
    # identity-poisoning guard — it must stay 1.0.
    name_emitted = 0
    name_correct = 0
    for m in rep["matches"]:
        pr = pred[m["pred_index"]]
        gr = gt[m["gt_index"]]
        pn, gn = _norm(pr.get("product_name", "")), _norm(gr.get("product_name", ""))
        if pn and pn.casefold() != "нет" and gn and gn.casefold() != "нет":
            name_emitted += 1
            if _similarity(pn, gn) >= 0.85:
                name_correct += 1

    out = {
        "eval": {
            k: rep[k]
            for k in (
                "pred_rows",
                "gt_rows",
                "matched_rows",
                "correct_rows",
                "score",
                "mean_field_accuracy",
                "mean_iou",
                "unmatched_predictions",
                "unmatched_ground_truth",
            )
        },
        "recall": {
            "gt_total": len(gt),
            "gt_geometrically_reachable": geo_recall,
            "pred_rows_with_bbox": len(p_boxes),
        },
        "name_precision": {
            "rows_emitting_name_on_match": name_emitted,
            "correct": name_correct,
            "precision": (
                round(name_correct / name_emitted, 4) if name_emitted else None
            ),
        },
    }

    if args.side_car and args.side_car.exists():
        sc = json.loads(args.side_car.read_text(encoding="utf-8"))
        rows = sc.get("rows", [])
        tiers: dict[str, int] = {}
        for r in rows:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        confs = [r["confidence"] for r in rows]
        out["confidence"] = {
            "rows": len(rows),
            "mode": sc.get("mode"),
            "tiers": tiers,
            "mean": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "ge_0.6": sum(1 for c in confs if c >= 0.6),
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
