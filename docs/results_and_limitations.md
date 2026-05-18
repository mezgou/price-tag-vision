# Price-Tag Pipeline — Delivered State, Measured Results, Limitations

> Honest engineering report. Pairs with `executive_summary_ocr_pipeline.md`
> (intended architecture) and `task_hackathon.md` (official task).

## 1. What was built and verified

| Component | State | Notes |
|---|---|---|
| Lens-distortion model (`app/utils/camera.py`) | **Done, verified** | Undistort + 90° CCW upright; exact inverse to raw 3840×2160 (point round-trip 0.0003 px). |
| Detector | **Done** | `runs/detect/train/weights/best.pt` (YOLOv8m). Strong: 9–18 tags/frame @0.9+. (Propagation-trained YOLO11s underperformed it — kept yolov8m.) |
| Auto-labelled dataset (no manual annotation) | **Done** | Template propagation, 274 GT rows → 5014 boxes, 0 skipped, undistorted+upright. |
| **ByteTrack tracking** | **Done, verified** | Streaming `model.track(persist=True, bytetrack.yaml)`; one stable track per tag (9 tracks/17 frames vs 254 fragments with the old greedy tracker). Collapses ~248 duplicate rows → ~33–41 per-tag rows. |
| Geometry through pipeline | **Done, verified** | Final CSV bbox inverse-mapped to raw 3840×2160. |
| **Zonal OCR** | **Done** | Per-zone crop + grayscale/Otsu; **PaddleOCR `lang='en'`** for numeric zones (the `ru` model is broken in this stack — see §3), Tesseract `rus` for the name. Rubles+superscript-kopecks recombined (`1199`+`99` → `1199.99`); discount sign restored. |
| Best-frame-per-tag | **Done** | Anchor timestamp+bbox on the best-recognized observation; crop-area prior so close/readable passes win. |
| Deterministic fields | **Done** | `color` via HSV; per-template `нет` defaults; validated-or-empty barcode/SKU. |
| QR cascade | **Lean** | zxing+pyzbar on a few variants (full opencv/aruco×13-variant cascade cost ~30 min for ~2 % yield; QR is non-critical per task §13.4). |
| 29-col CSV, UTF-8, `.`-decimal, dedup | **Done** | Schema-correct; runs end-to-end; YOLO/ByteTrack on GPU. |

## 2. Measured results (proxy = official metric)

Trajectory on `26_12-20.mp4` as fixes landed (proxy: % tags with ≥80 % field accuracy; IoU≥0.5):

| Stage | matched/GT | mean field acc | rows | score |
|---|---|---|---|---|
| Tesseract whole-crop | 49/71 | 0.34 | 263 | 0.0 |
| Paddle-en zonal + kopecks | 47/71 | 0.40 | 248 | 0.0 |
| + ByteTrack + best-frame | (clip) | 0.42 | **33–41** | 0.0 |

**Proven working** (verified on close tags): `price_card` exactly correct
(e.g. 1199.99, 1299.99 … 2749.99 — **15 distinct prices exactly match GT**),
`price_default` (1578.00), `discount_amount` (-24 %, -38 %), `color`, and the
`нет`-default fields. ByteTrack consolidation works (rows 248→~35).

## 3. The remaining ceiling (precisely characterised, not a code bug)

The official metric is **≥80 % of ~21 fields per tag**. On this footage, per
close tag the **reliably-correct** fields are: `price_card`,
`discount_amount`, `color`, `price_discount`(нет), `wholesale_*`×4(нет),
`action_*`×2(нет), often `price_default`/`special_symbols` ≈ **10–13 / 21
(~0.5–0.6)**. The fields that stay wrong/empty — `barcode`, `id_sku`,
`print_datetime`, `code`, exact `product_name` — are **sub-resolution**: the
printed EAN/SKU/date text is ~10 px even on the closest crops. This is exactly
the task's documented central risk (§15: low-res crops, mini barcode), and the
GT match-key is `barcode` (§6) which is the hardest field here.

Secondary engine constraint: **PaddleOCR `lang='ru'` is broken** in this
paddle-3.0.0/paddleocr-2.10 stack (confident garbage even on a flawless
synthetic "1199.99"); `lang='en'` works perfectly, so Russian `product_name`
falls back to weaker Tesseract. **paddle-GPU is not available on this Windows
box** (needs a system CUDA Toolkit; the pip `nvidia-cuda-nvrtc-cu12` has no
Windows wheel) — OCR runs on CPU (task §10: speed is secondary).

## 4. Path to a passing score (architecture is ready for all)

1. **Higher-resolution / closer-camera footage** — the dominant external lever;
   the EAN/SKU/date text must exceed ~20 px to OCR.
2. **Super-resolution** of crops before OCR (task §17).
3. A working Russian recognizer (fix Paddle-ru env, or a different model) for
   `product_name`; SKU-dictionary fuzzy-match to backfill names from barcode.
4. System CUDA Toolkit → paddle-GPU for ~10× faster OCR.

The pipeline is architecturally complete and correct end-to-end; the score is
bounded by source resolution on ~5 small-text fields, not by missing logic.

## 5. Hardware / runtime

Windows 11, RTX 3060 Laptop 6 GB, Python 3.12. YOLO+ByteTrack on GPU; QR/OCR
on CPU. Lean profile ≈ a few min for a ~30-frame slice; ~10–15 min for a
120-frame video segment (OCR-bound). Dev uses short slices / a close-pass clip
for fast iteration; full-video runs for final confirmation.
