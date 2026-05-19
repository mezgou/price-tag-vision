"""Does a dedicated high-upscale name-band OCR pass lift catalog recall
(the only recoverable score lever) WITHOUT losing 100% name precision?
Close clip, top-K crops/track, full-crop tokens vs full-crop+nameband.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

from app.pipelines.price_tag_v5.catalog import CatalogResolver
from app.pipelines.price_tag_v5.stages.block_ocr_catalog import (
    V5BlockOcrCatalogStage,
)
from app.utils.camera import CameraModel

CLIP = R / "artifacts/clip_close.mp4"
WEIGHTS = R / "runs/detect/train/weights/best.pt"
K, MAX_TRACKS = 6, 14


def _sharp(i):
    return float(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def main() -> int:
    cm, model = CameraModel(), YOLO(str(WEIGHTS))
    bank: dict[int, list] = defaultdict(list)
    cap = cv2.VideoCapture(str(CLIP))
    i = 0
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

    gt_bc, gt_names = set(), []
    for row in csv.DictReader(open(R / "data/videos/26_12-20/26_12-20.csv",
                                   encoding="utf-8")):
        b = re.sub(r"\D", "", row.get("barcode") or "")
        if len(b) in (12, 13):
            gt_bc.add(b)
        nm = (row.get("product_name") or "").strip()
        if nm and nm.lower() != "нет":
            gt_names.append(nm)

    from paddleocr import PaddleOCR
    st = V5BlockOcrCatalogStage()
    st._engine = PaddleOCR(lang="en", use_textline_orientation=True)
    rc = CatalogResolver()

    def score(pool_fn, label):
        acc = nh = bcgt = 0
        t0 = time.time()
        for tid, crops in tracks:
            pooled = []
            for _, cr in crops:
                pooled += pool_fn(cr)
            m = rc.resolve(pooled, category="wine")
            if not m or not m.accepted:
                continue
            acc += 1
            if any(m.product_name[:22].casefold() in g.casefold()
                   or g[:22].casefold() in m.product_name.casefold()
                   for g in gt_names):
                nh += 1
            if m.barcode in gt_bc:
                bcgt += 1
        print(f"{label:26} accepted={acc:2}/{len(tracks)} "
              f"name∈GT={nh}/{acc or 1} bc∈GT={bcgt}/{acc or 1} "
              f"{(time.time()-t0):.0f}s")

    score(lambda cr: [b["text"] for b in st._ocr_blocks(cr, 3.0)],
          "full-crop only (base)")
    score(lambda cr: [b["text"] for b in st._ocr_blocks(cr, 3.0)]
          + st._namestrip_texts(cr),
          "full-crop + name-band")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
