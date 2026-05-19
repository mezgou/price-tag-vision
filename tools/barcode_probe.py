"""Barcode-strip feasibility probe (Priority 1).

Question: on the close clip, can we VISUALLY decode the printed EAN barcode
(zxing/pyzbar/cv2) or OCR enough digits to disambiguate the catalog's
same-name EAN candidates? Pure-catalog name->barcode is ~45% (every GT
fullname maps to >1 EAN), so a real per-tag barcode signal is the only
honest way to lift `barcode` recall. Fast, isolated, no full pipeline.

clip -> YOLOv8m+ByteTrack -> top-K crops/track -> bottom-strip multi-scale
decode + digit OCR -> per-track barcode -> precision vs 26_12-20 GT.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch  # noqa: F401  # before paddleocr (albumentations DLL)

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

from app.pipelines.price_tag_v5.catalog import ean13_ok, to_ean13
from app.utils.camera import CameraModel

CLIP = R / "artifacts/clip_close.mp4"
WEIGHTS = R / "runs/detect/train/weights/best.pt"
K = 6
MAX_TRACKS = 14


def _sharp(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def _zx():
    try:
        import zxingcpp
        return zxingcpp
    except Exception:
        return None


def _pyz():
    try:
        from pyzbar.pyzbar import decode as d
        return d
    except Exception:
        return None


ZX = _zx()
PYZ = _pyz()


def _norm_ean(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")
    if len(d) in (12, 13, 14):
        e = to_ean13(d)
        if e and ean13_ok(e):
            return e
    return ""


def _decode_variants(img: np.ndarray) -> list[str]:
    """All barcode hits across strip/scale/threshold variants."""
    h, w = img.shape[:2]
    out: list[str] = []
    regions = [img, img[int(h * 0.70):h, :], img[int(h * 0.78):h, :]]
    for reg in regions:
        if reg.size == 0:
            continue
        rh, rw = reg.shape[:2]
        for fx in (3, 5, 8):
            big = cv2.resize(reg, (rw * fx, rh * fx),
                             interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            otsu = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            sharp = cv2.addWeighted(
                gray, 1.6,
                cv2.GaussianBlur(gray, (0, 0), 3), -0.6, 0)
            for v in (gray, otsu, sharp):
                if ZX is not None:
                    try:
                        for r in ZX.read_barcodes(v):
                            e = _norm_ean(r.text)
                            if e:
                                out.append(e)
                    except Exception:
                        pass
                if PYZ is not None:
                    try:
                        for r in PYZ(v):
                            e = _norm_ean(r.data.decode("ascii", "ignore"))
                            if e:
                                out.append(e)
                    except Exception:
                        pass
    return out


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
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = (int(v) for v in box)
                if x2 - x1 < 70 or y2 - y1 < 70:
                    continue
                cr = proc[max(0, y1):y2, max(0, x1):x2]
                if cr.size:
                    s = _sharp(cr) * np.sqrt(cr.shape[0] * cr.shape[1])
                    b = bank[int(tid)]
                    b.append((s, cr.copy()))
                    b.sort(key=lambda z: -z[0])
                    del b[K:]
        i += 1
    cap.release()
    tracks = sorted(bank.items(), key=lambda kv: -kv[1][0][0])[:MAX_TRACKS]
    print(f"bytetrack {len(bank)} tracks {i} frames {time.time()-t0:.0f}s; "
          f"decode top {len(tracks)} x{K}; zxing={ZX is not None} "
          f"pyzbar={PYZ is not None}")

    gt_bc = set()
    for row in csv.DictReader(open(R / "data/videos/26_12-20/26_12-20.csv",
                                   encoding="utf-8")):
        b = re.sub(r"\D", "", row.get("barcode") or "")
        if len(b) in (12, 13):
            gt_bc.add(b)

    t1 = time.time()
    decoded = in_gt = 0
    for tid, crops in tracks:
        votes: Counter[str] = Counter()
        for _, cr in crops:
            for e in _decode_variants(cr):
                votes[e] += 1
        if not votes:
            print(f"trk{tid:>3} x{len(crops)}  no-decode")
            continue
        best, n = votes.most_common(1)[0]
        decoded += 1
        ok = best in gt_bc
        in_gt += ok
        print(f"trk{tid:>3} x{len(crops)}  {best}  votes={n} "
              f"{'∈GT' if ok else 'NOT-IN-GT'}")
    n = len(tracks)
    print(f"\ndecoded={decoded}/{n}  barcode∈GT={in_gt}  "
          f"decode {time.time()-t1:.0f}s")
    print(f"VISUAL-DECODE precision = {(in_gt/decoded if decoded else 0):.0%}"
          f"  recall = {in_gt}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
