"""Fine-print feasibility probe: can we recover id_sku / print_datetime /
code from the close clip via a high-upscale bottom-strip pass + multi-crop
consensus? These 3 fields gate the metric (oracle: 0.014 -> 0.96).

clip -> YOLO+ByteTrack -> top-K crops/track -> _fineprint_texts + main OCR
-> consensus -> compare to 26_12-20 GT. Fast, no full pipeline.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401  # before paddleocr

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

from app.pipelines.price_tag_v5.stages.block_ocr_catalog import (
    V5BlockOcrCatalogStage,
)
from app.utils.camera import CameraModel

CLIP = R / "artifacts/clip_close.mp4"
WEIGHTS = R / "runs/detect/train/weights/best.pt"
K = 6
MAX_TRACKS = 14


def _sharp(i):
    return float(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def main() -> int:
    cm = CameraModel()
    model = YOLO(str(WEIGHTS))
    bank: dict[int, list] = defaultdict(list)
    cap = cv2.VideoCapture(str(CLIP))
    i = 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 2:
            i += 1
            continue
        proc = cm.raw_frame_to_processed(fr)
        res = model.track(proc, persist=True, tracker="bytetrack.yaml",
                          imgsz=1280, conf=0.35, iou=0.5, device=0,
                          verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            for b, t in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = (int(v) for v in b)
                if x2 - x1 < 70 or y2 - y1 < 70:
                    continue
                cr = proc[max(0, y1):y2, max(0, x1):x2]
                if cr.size:
                    s = _sharp(cr) * np.sqrt(cr.shape[0] * cr.shape[1])
                    lst = bank[int(t)]
                    lst.append((s, cr.copy()))
                    lst.sort(key=lambda z: -z[0])
                    del lst[K:]
        i += 1
    cap.release()
    tracks = sorted(bank.items(), key=lambda kv: -kv[1][0][0])[:MAX_TRACKS]
    print(f"bytetrack {len(bank)} tracks {i} fr {time.time()-t0:.0f}s; "
          f"fine-print top {len(tracks)} x{K}")

    gt_sku, gt_dt, gt_code = set(), set(), set()
    for row in csv.DictReader(open(R / "data/videos/26_12-20/26_12-20.csv",
                                   encoding="utf-8")):
        s = re.sub(r"\D", "", row.get("id_sku") or "")
        if len(s) == 12:
            gt_sku.add(s)
        d = (row.get("print_datetime") or "").strip()
        if d and d != "нет":
            gt_dt.add(d)
        c = (row.get("code") or "").strip()
        if c and c != "нет":
            gt_code.add(c)

    from paddleocr import PaddleOCR
    st = V5BlockOcrCatalogStage()
    st._engine = PaddleOCR(lang="en", use_textline_orientation=True)

    t1 = time.time()
    sku_hit = sku_set = dt_set = code_set = 0
    for tid, crops in tracks:
        skus, dates, codes = [], [], []
        for _, cr in crops:
            blocks = st._ocr_blocks(cr, 3.0)
            texts = [b["text"] for b in blocks] + st._fineprint_texts(cr)
            skus += st._sku_candidates(texts)
            dates += st._date_candidates(texts)
            codes += st._code_candidates(texts)
        sku = st._consensus_sku(skus)
        dt = st._mode(dates)
        code = st._mode(codes)
        sku_set += bool(sku)
        dt_set += bool(dt)
        code_set += bool(code)
        ok = sku in gt_sku and sku
        sku_hit += ok
        print(f"trk{tid:>3} sku={sku or '-':<13}"
              f"{'∈GT' if ok else '   '} dt={dt or '-':<18} "
              f"code={code or '-'}")
    n = len(tracks)
    print(f"\nsku set={sku_set}/{n} exact∈GT={sku_hit}  "
          f"dt set={dt_set}/{n}  code set={code_set}/{n}  "
          f"({time.time()-t1:.0f}s)")
    print(f"GT pool: sku={len(gt_sku)} dt={len(gt_dt)} code={len(gt_code)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
