"""Standalone OCR module probe — NOT part of any pipeline.

Goal: isolate and *see* what OCR gives us on real ByteTrack crops, fast.
ByteTrack is proven to work, so:
  clip -> YOLOv8m+ByteTrack -> best raw crop per track
       -> PaddleOCR(lang=en) FULL detect+recognize on the whole crop
       -> per text block: text, conf, rel-height -> geometric class
       -> dump JSON + annotated debug image per track

This tests the key borrowed idea: let Paddle's own DB detector find text
blocks (instead of fixed bleeding zones) and classify by block height.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401  # MUST precede paddleocr (albumentations DLL)

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))

import cv2
import numpy as np
from ultralytics import YOLO

from app.utils.camera import CameraModel

CLIP = R / "artifacts/clip_close.mp4"
OUT = R / "artifacts/ocr_probe"
WEIGHTS = R / "runs/detect/train/weights/best.pt"
MAX_TRACKS = 12


def _sharp(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _classify(rel_h: float, rel_y: float) -> str:
    if rel_h >= 0.12:
        return "PRICE"
    if rel_h >= 0.05:
        return "NAME" if rel_y < 0.45 else "MID"
    return "FINE"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cm = CameraModel()
    model = YOLO(str(WEIGHTS))

    # 1. ByteTrack over the clip -> collect best raw crop per track
    best: dict[int, tuple[float, np.ndarray]] = {}
    cap = cv2.VideoCapture(str(CLIP))
    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 2:  # ~2x temporal subsample, ByteTrack still stable
            idx += 1
            continue
        proc = cm.raw_frame_to_processed(frame)
        res = model.track(
            proc, persist=True, tracker="bytetrack.yaml",
            imgsz=1280, conf=0.35, iou=0.5, device=0, verbose=False,
        )[0]
        if res.boxes is None or res.boxes.id is None:
            idx += 1
            continue
        for box, tid in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 - x1 < 60 or y2 - y1 < 60:
                continue
            crop = proc[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            score = _sharp(crop) * np.sqrt(crop.shape[0] * crop.shape[1])
            tid = int(tid)
            if tid not in best or score > best[tid][0]:
                best[tid] = (score, crop.copy())
        idx += 1
    cap.release()
    track_secs = time.time() - t0
    tracks = sorted(best.items(), key=lambda kv: -kv[1][0])[:MAX_TRACKS]
    print(f"bytetrack: {len(best)} tracks, {idx} frames, {track_secs:.1f}s; "
          f"probing top {len(tracks)} crops")

    # 2. PaddleOCR-en full detect+recognize on each best crop
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)

    report = []
    t1 = time.time()
    for tid, (score, crop) in tracks:
        h, w = crop.shape[:2]
        up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # API differs by build: prefer .predict (3.x dict), fall back to
        # legacy .ocr (list of [poly,(text,score)]).
        items: list[tuple[np.ndarray, str, float]] = []
        if hasattr(ocr, "predict"):
            res = ocr.predict(up)
            r0 = res[0] if isinstance(res, list) else res
            if isinstance(r0, dict):
                tx = r0.get("rec_texts", []) or []
                sc = r0.get("rec_scores", []) or []
                pl = r0.get("rec_polys") or r0.get("dt_polys") or []
                for i, t in enumerate(tx):
                    if i < len(pl):
                        items.append((np.array(pl[i]), t,
                                      float(sc[i]) if i < len(sc) else 0.0))
        else:
            res = ocr.ocr(up)
            page = res[0] if res else []
            for it in page or []:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    poly, payload = it[0], it[1]
                    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                        items.append((np.array(poly), str(payload[0]),
                                      float(payload[1])))
        blocks = []
        vis = up.copy()
        for poly, txt, sc_i in items:
            if poly is None or poly.size == 0:
                continue
            xs, ys = poly[:, 0], poly[:, 1]
            bh = float(ys.max() - ys.min())
            bw = float(xs.max() - xs.min())
            rel_h = bh / float(up.shape[0])
            rel_y = float(ys.min()) / float(up.shape[0])
            cls = _classify(rel_h, rel_y)
            blocks.append({
                "text": txt,
                "conf": round(float(sc_i), 3),
                "cls": cls,
                "rel_h": round(rel_h, 3),
                "rel_y": round(rel_y, 3),
                "aspect": round(bw / max(bh, 1), 1),
            })
            cv2.polylines(vis, [poly.astype(int)], True, (0, 0, 255), 2)
            cv2.putText(vis, f"{cls}:{txt[:14]}", (int(xs.min()), int(ys.min()) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        cv2.imwrite(str(OUT / f"trk{tid:03d}_{w}x{h}.jpg"),
                    cv2.resize(vis, None, fx=0.5, fy=0.5))
        report.append({"track": tid, "crop": f"{w}x{h}",
                       "n_blocks": len(blocks), "blocks": blocks})
        print(f"trk{tid:03d} {w}x{h}: {len(blocks)} blocks | "
              + " | ".join(f"{b['cls']}={b['text']}" for b in blocks[:6]))
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ocr: {time.time() - t1:.1f}s for {len(tracks)} crops -> "
          f"{OUT}/report.json + annotated jpgs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
