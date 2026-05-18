"""Multi-frame token-pool probe: does pooling brand tokens across a track's
top-K crops raise CatalogResolver recall (vs the 1-crop baseline 3/12)
while keeping 100% name precision? Fast, isolated, no pipeline.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401  # before paddleocr (albumentations DLL)

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

from app.pipelines.price_tag_v5.catalog import CatalogResolver
from app.utils.camera import CameraModel

CLIP = R / "artifacts/clip_close.mp4"
WEIGHTS = R / "runs/detect/train/weights/best.pt"
K = 5            # top crops per track to OCR + pool
MAX_TRACKS = 14


def _sharp(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def main() -> int:
    cm = CameraModel()
    model = YOLO(str(WEIGHTS))
    # per track: heap-ish list of (score, crop)
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
        if res.boxes is None or res.boxes.id is None:
            i += 1
            continue
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.id.cpu().numpy()):
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 - x1 < 70 or y2 - y1 < 70:
                continue
            crop = proc[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            s = _sharp(crop) * np.sqrt(crop.shape[0] * crop.shape[1])
            b = bank[int(tid)]
            b.append((s, crop.copy()))
            b.sort(key=lambda z: -z[0])
            del b[K:]
        i += 1
    cap.release()
    tracks = sorted(bank.items(), key=lambda kv: -kv[1][0][0])[:MAX_TRACKS]
    print(f"bytetrack {len(bank)} tracks {i} frames {time.time()-t0:.1f}s; "
          f"pooling top {len(tracks)} tracks x{K} crops")

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)

    gt_bc, gt_names = set(), []
    for row in csv.DictReader(open(R / "data/videos/26_12-20/26_12-20.csv",
                                   encoding="utf-8")):
        bc = re.sub(r"\D", "", row.get("barcode") or "")
        if len(bc) in (12, 13):
            gt_bc.add(bc)
        nm = (row.get("product_name") or "").strip()
        if nm and nm.lower() != "нет":
            gt_names.append(nm)
    res_cat = CatalogResolver()

    t1 = time.time()
    acc = name_hit = bc_set = bc_ok = 0
    for tid, crops in tracks:
        pooled: list[str] = []
        digit_hint = ""
        for _, crop in crops:
            up = cv2.resize(crop, None, fx=3, fy=3,
                            interpolation=cv2.INTER_CUBIC)
            r = ocr.ocr(up)
            page = r[0] if r else []
            for it in page or []:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    payload = it[1]
                    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                        txt = str(payload[0])
                        pooled.append(txt)
                        d = re.sub(r"\D", "", txt)
                        if len(d) in (12, 13) and not digit_hint:
                            digit_hint = d
        m = res_cat.resolve(pooled, category="wine", barcode_hint=digit_hint)
        if m is None or not m.accepted:
            print(f"trk{tid:>3} x{len(crops)}  —")
            continue
        acc += 1
        nmatch = any(m.product_name[:22].casefold() in g.casefold()
                     or g[:22].casefold() in m.product_name.casefold()
                     for g in gt_names)
        name_hit += nmatch
        if m.barcode:
            bc_set += 1
            bc_ok += m.barcode in gt_bc
        print(f"trk{tid:>3} x{len(crops)} ACC bc={m.barcode or '-':<14}"
              f"{'∈GT' if m.barcode in gt_bc else '   '} "
              f"{'NAME✓' if nmatch else 'NAME✗'} {m.product_name[:48]}")
    n = len(tracks)
    print(f"\npooled: accepted={acc}/{n}  name✓={name_hit}/{acc if acc else 1}"
          f"  barcode_set={bc_set} barcode∈GT={bc_ok}"
          f"  ocr {time.time()-t1:.1f}s")
    print(f"NAME precision={name_hit}/{acc} "
          f"({(name_hit/acc if acc else 0):.0%}); "
          f"vs 1-crop baseline accepted=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
