"""Visual showcase: how v5 detects + recognizes tags.

Produces, in artifacts/v5_showcase/:
  - frame_overlay.jpg : one processed frame with all ByteTrack boxes+IDs
  - trkXXX.jpg        : per-track best crop annotated with parsed price /
                        discount / catalog product name
  - contact_sheet.jpg : montage of annotated crops
Fast: close clip only, top crops, one OCR pass each.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch  # noqa: F401  # before paddleocr

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))

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
OUT = R / "artifacts/v5_showcase"
TOPK = 4
MAX_TRACKS = 12


def _sharp(i):
    return float(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cm, model = CameraModel(), YOLO(str(WEIGHTS))
    bank: dict[int, list] = defaultdict(list)
    overlay = None
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
            n = len(res.boxes.id)
            if overlay is None or n >= 6:
                ov = proc.copy()
                for b, t in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.id.cpu().numpy()):
                    x1, y1, x2, y2 = (int(v) for v in b)
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 200, 0), 3)
                    cv2.putText(ov, f"#{int(t)}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
                overlay = (n, ov)
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
                    del lst[TOPK:]
        i += 1
    cap.release()
    if overlay is not None:
        sc = 1400.0 / max(overlay[1].shape[:2])
        cv2.imwrite(str(OUT / "frame_overlay.jpg"),
                    cv2.resize(overlay[1], None, fx=sc, fy=sc),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    tracks = sorted(bank.items(), key=lambda kv: -kv[1][0][0])[:MAX_TRACKS]
    print(f"bytetrack {len(bank)} tracks {i} fr {time.time()-t0:.0f}s; "
          f"showcasing {len(tracks)}")

    from paddleocr import PaddleOCR
    stage = V5BlockOcrCatalogStage()
    stage._engine = PaddleOCR(lang="en", use_textline_orientation=True)
    res_cat = CatalogResolver()

    cells = []
    for tid, crops in tracks:
        pooled, hint = [], ""
        fields = {}
        best_crop = crops[0][1]
        for _, cr in crops:
            blk = stage._ocr_blocks(cr, 3.0)
            if not blk:
                continue
            pooled += [b["text"] for b in blk]
            if not fields:
                fields = stage._prices_from_blocks(blk)
        m = res_cat.resolve(pooled, category="wine", barcode_hint=hint)
        name = m.product_name if (m and m.accepted) else ""
        big = cv2.resize(best_crop, (360, 460))
        panel = np.full((460, 760, 3), 248, np.uint8)
        panel[:, :360] = big
        y = 40

        def line(txt, col=(20, 20, 20)):
            nonlocal y
            cv2.putText(panel, txt, (374, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, col, 2)
            y += 34
        line(f"track #{tid}", (0, 120, 0))
        line(f"price_card:    {fields.get('price_card', '-')}")
        line(f"price_default: {fields.get('price_default', '-')}")
        line(f"discount:      {fields.get('discount_amount', '-')}")
        line(f"special:       {fields.get('special_symbols', '-')}")
        line(f"color:         {fields.get('color', '-')}")
        ok_id = bool(name)
        line("catalog id: " + ("ACCEPTED" if ok_id else "withheld"),
             (0, 140, 0) if ok_id else (0, 0, 200))
        # wrap long name
        nm = name or "(not confidently identified)"
        for k in range(0, len(nm), 30):
            line(nm[k:k + 30], (90, 60, 0))
        cv2.imwrite(str(OUT / f"trk{tid:03d}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        cells.append(panel)

    if cells:
        cols = 3
        rows = (len(cells) + cols - 1) // cols
        H, W = 460, 760
        sheet = np.full((rows * H, cols * W, 3), 255, np.uint8)
        for idx, c in enumerate(cells):
            r_, cc = divmod(idx, cols)
            sheet[r_ * H:r_ * H + H, cc * W:cc * W + W] = c
        cv2.imwrite(str(OUT / "contact_sheet.jpg"),
                    cv2.resize(sheet, None, fx=0.6, fy=0.6),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {OUT}: frame_overlay.jpg, contact_sheet.jpg, "
          f"{len(cells)} trk*.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
