# Agent Brief — Build `price_tag_v3`: a track-centric multi-view reconstruction pipeline

You are an autonomous engineering agent working in the repo
`price-tag-vision`. Your job: design and build a **new, separate** video
pipeline `price_tag_v3` that beats `price_tag_v2` on the official metric, by
**creatively escaping the resolution wall that capped v2 at score 0.0**.

Read this whole brief before acting. It encodes everything an earlier effort
learned the hard way — do not relearn it by trial and error.

---

## 0. Hard rules

1. **Do NOT modify or break `price_tag_v2`, `price_tag_cpu_v1`, the backend,
   frontend, or the eval harness contract.** `v3` is a NEW pipeline package
   `vision_service/app/pipelines/price_tag_v3/`, registered as a new pipeline
   name, selectable independently. v2 must keep running exactly as today.
2. **Reuse, don't fork**: import v2/v1 building blocks you can reuse
   (`app/utils/camera.py` CameraModel, FrameSampling, the eval module, the
   dataset builder, the YOLOv8m detector at `runs/detect/train/weights/best.pt`).
3. Local only, no cloud APIs. Python via `uv run --project vision_service`.
4. Keep a **fast iteration harness** at every step (see §6). OCR is the time
   sink; never debug on the full video.
5. Be honest in measurements. The official metric is: **% of GT tags whose
   matched row has ≥80% correct fields**, match key = `barcode` first then
   spatial (this is already implemented faithfully in
   `app/evaluation/price_tag_eval.py` — reuse it).

---

## 1. What is already proven (do not re-investigate)

**Works and is verified:**
- Camera geometry: `app/utils/camera.py::CameraModel` — lens-undistort + 90°
  CCW upright, exact inverse to raw 3840×2160 (point round-trip 0.0003 px).
  Reuse as-is. All processing happens in undistorted+upright space; the final
  CSV bbox must be mapped back with `processed_box_to_raw`.
- Detector: `runs/detect/train/weights/best.pt` (YOLOv8m), strong: 9–18
  tags/frame @0.9+ conf. Trained on a 66-image manual set; a
  template-propagation auto-dataset (`app/datasets/yolo_price_tag_dataset.py`,
  `--undistort`) exists but its YOLO11s underperformed the yolov8m — **keep
  yolov8m as the detector**.
- ByteTrack: `model.track(persist=True, tracker='bytetrack.yaml')` over
  temporally-ordered undistorted frames gives **one stable track per physical
  tag** (≈9 tracks / 17 frames; collapses 248 dup rows → ~70). This is the
  backbone of the new approach — see §3.
- PaddleOCR **`lang='en'` works** and is correct on numeric content
  (synthetic "1199.99" ✓, real close-crop price ✓). **`lang='ru'` is BROKEN**
  in this stack (confident garbage even on flawless digits). Use Paddle-en
  for numbers; Tesseract `rus` (weak) or a better RU model for Cyrillic.
- Price tags show big rubles + tiny superscript kopecks; OCR returns them as
  separate tokens. Recombine to `RUB.KK` (logic in
  `price_tag_v2/stages/zonal_ocr.py::_best_price`). Discounts lose their minus
  → re-add (`_best_discount`). Reuse this parsing logic.
- **GT-verified field relationships (across all 274 labelled rows):**
  `qr_code_barcode == barcode` 99%, `price1_qr == price_default` 97%,
  `price4_qr == price_card` 97%, `price_discount == нет` 100%,
  `action_*`/`wholesale_l1_price`/`wholesale_l2_* == нет` ~98%,
  `price3_qr == нет` 73%, `wholesale_l1_count == нет` 62%, `price2_qr` is a
  distinct price (NOT == price_default). v2 already backfills these
  bidirectionally — reuse and rely on it.

**Proven dead ends (do NOT repeat):**
- Single-best-frame OCR: even the closest crop is ~200–450 px; barcode/SKU/
  date/code text is ~10–15 px → unreadable by Tesseract AND Paddle. This is
  THE wall. v2 plateaued at mean field accuracy ~0.42, score 0.0.
- QR decoding: ~2% of crops at this camera distance (zxing/pyzbar). Keep as a
  cheap bonus only; QR is non-critical per the task.
- cv2 `dnn_superres` and `wechat_qrcode` are **empty stubs** in this
  opencv-contrib-headless build.
- paddle-GPU on Windows: needs a system CUDA toolkit (pip
  `nvidia-cuda-nvrtc-cu12` has no Windows wheel). OCR stays on CPU; that's
  acceptable (task: speed secondary, ~30–50 s/tag budget).

---

## 2. The core idea — turn the weakness into the strength

A moving robot films **each tag 20–40 times** from slightly different
distances, angles and sub-pixel offsets. v2 threw this away (it OCR'd the one
"best" crop). **A camera that passes a tag many times is a free multi-view
capture rig.** Use it:

> **Track-centric multi-view reconstruction**: for each ByteTrack track,
> collect ALL its crops, geometrically register them, fuse them into a single
> **super-resolved, deblurred, rectified** tag image, then decode that — with
> cross-frame consensus and catalog constraints as error-correction.

This attacks the resolution wall with information physics, not better single
-frame OCR.

---

## 3. Required architecture (`price_tag_v3`)

Stages (each a `BaseStage`, fail-soft, in `price_tag_v3/stages/`):

1. **frame_sampling** — reuse v2's; undistort+upright via CameraModel.
   Add a *motion/stop detector* (optical-flow magnitude): sample densely
   during the closest-approach / slow segments of each tag, sparse elsewhere.
2. **yolo_bytetrack** — reuse v2's `YoloByteTrackStage` (yolov8m, persist).
   One stable `track_id` per tag.
3. **track_crop_bank** — per track, keep ALL crops (not top-K), with
   sharpness, area, glare, estimated motion-blur, and the homography needed to
   warp each to a common canonical tag frame.
4. **multi_view_fusion** *(the heart — new)*. Per track:
   - Select the K sharpest, largest, least-glare crops (e.g. K=8–20).
   - **Rectify**: estimate the tag quadrilateral (edges/contour or the QR/
     finder squares or the bounding rect of the white+red regions) and
     homography-warp every selected crop to a fixed canonical size
     (e.g. 900×1200), so all views are pixel-aligned.
   - **Sub-pixel register** the rectified views to a reference (ECC /
     phase-correlation).
   - **Fuse → super-resolution**: robust temporal fusion (median / Wiener /
     simple shift-and-add upsampling, or a learned MFSR if you can run one on
     GPU via torch — torch CUDA works here). Output one high-SNR,
     higher-effective-resolution canonical tag image per track.
   - Also produce a deblurred variant (Richardson–Lucy / Wiener with an
     estimated PSF) for the motion-blurred tracks.
5. **zonal_decode** — on the fused canonical image (zones are now in fixed
   canonical coordinates → reliable):
   - Numeric zones (price_default, price_card, discount, barcode digits, SKU,
     date) → PaddleOCR **lang='en'** + Tesseract digit fallback. Reuse v2's
     `_best_price` / `_best_discount` / EAN-13 logic.
   - **Barcode from the BARS, not the digits** *(new, high value)*: the EAN-13
     bar pattern is a 1-D signal. On the fused image, take the barcode-zone
     band, average many scan-lines (multi-row + the fused multi-frame SNR),
     binarize, and decode with zxing/pyzbar/zbar in 1-D barcode mode. Bars
     survive low resolution far better than the printed digits. This is the
     biggest single opportunity v2 missed.
   - Cyrillic name zone → best available RU recognizer (try fixing PaddleOCR
     PP-OCRv5 `ru`, or EasyOCR `ru`, or a small fine-tuned CRNN on synthetic
     price-tag fonts; Tesseract `rus` as floor).
6. **char_consensus** — for digit fields, OCR several fused/per-frame variants
   and do **character-position-level voting** constrained by structure:
   EAN-13 checksum (brute-force ≤2 uncertain positions to satisfy it), price
   ends in a common kopeck (.99/.90/.49…), date regex `DD.MM.YYYY[ HH:MM]`,
   SKU = 12 digits. Consensus + constraints recover strings no single frame
   gets right.
7. **catalog_resolver** *(new, high value, task-legal)*. Build a reference
   catalog from the **provided labelled CSVs** (`data/videos/*/*.csv` are a
   barcode↔id_sku↔product_name↔price table for these exact products) — and
   optionally an offline scrape if you add one with a clear source. For each
   track, fuzzy-match the confidently-read fields (partial name + a correct
   price + color) to the catalog; if a single entry is a strong match,
   **backfill barcode / id_sku / exact product_name from the catalog**. This
   recovers the sub-resolution fields *without* reading them, and barcode is
   the GT match key, so this also unlocks matching. Be principled: only
   backfill when the match is high-confidence and price-consistent; never
   fabricate. Document this clearly (task §13.10/§13.12 allow own SKU bases
   with transparent description).
8. **row_fusion** — reuse v2's: per-track field majority vote, bidirectional
   QR↔visible backfill, per-template `нет` defaults, recognition-aware
   best-frame for timestamp+bbox, exact bbox→raw.
9. **csv_writer / preview** — reuse v2's; 29-col schema, sample.csv format.

---

## 4. Creative levers to try (ranked by expected value)

1. **Catalog resolver (§3.7)** — likely the single biggest win: turns a
   correctly-read price + partial name into a *guaranteed-correct*
   barcode/sku/name via lookup, unlocking both field accuracy and the
   barcode match key. Start here.
2. **Barcode-from-bars on the fused image (§3.5)** — recover the GT match key
   directly from the bar signal.
3. **Multi-view fusion / SR (§3.4)** — lifts every numeric field; rectified
   canonical zones also fix the zone-bleed that hurt v2.
4. **Character-consensus + checksum error-correction (§3.6)** — cheap, strong
   for barcode/price/date.
5. **A working RU recognizer** for product_name (helps the fuzzy catalog
   match too).
6. **Synthetic-font fine-tuned digit recognizer** if OCR engines still miss.

---

## 5. Constraints & environment gotchas (memorize)

- `uv run --project vision_service python ...` only. Background long jobs via
  the runner's background mode, not `nohup &`. Clean dirs with PowerShell
  `Remove-Item -Recurse -Force` (git-bash `rm -rf` hits "Directory not empty").
- Import `torch` BEFORE `paddleocr` (albumentations→torch DLL, WinError 127).
- `paddlepaddle==3.0.0` is pinned (3.3.1 crashes oneDNN). torch is cu130 GPU
  (works); paddle is CPU-only here.
- YOLO/ByteTrack on GPU (`device="0"`), OCR on CPU.
- Frame sampling already does undistort+upright; GT bbox/CSV must be raw
  3840×2160 (use `CameraModel.processed_box_to_raw`).
- Data: `data/videos/<name>/<name>.{mp4,csv}` (5 labelled + unlabeled). Tags
  are ~273/274 red, mostly the alcohol layout.

---

## 6. Fast iteration harness (build this first, use it always)

1. One-time: cut a ~16 s close-pass clip (frames ~680–1010 of
   `26_12-20.mp4`) → `artifacts/clip_close.mp4` (a close-up of the reference
   "Вино POGGIO TOSCO … 1199.99 -24%" tag is at frame ~780).
2. Iterate on the clip: `smoke_price_tag_pipeline.py --pipeline price_tag_v3
   --video artifacts/clip_close.mp4 --sample-fps 4 --max-frames 70` then the
   eval. A cycle must be a few minutes, never 30.
3. For unit feedback, dump intermediate fused-canonical images +
   per-zone OCR text to `artifacts/` and inspect them directly.
4. Only run the full `26_12-20.mp4` for a milestone score.

---

## 7. Success criteria (in order)

1. v3 runs end-to-end, produces a valid 29-col CSV, **v2 still works**.
2. On `26_12-20.mp4` via `eval_price_tag_predictions.py`: beat v2's baseline
   (`mean_field_accuracy 0.42`, `score 0.0`). Target ladder:
   - **A**: barcode correctly recovered (bars or catalog) on ≥30% of close
     tags → matched_rows and barcode field jump.
   - **B**: `mean_field_accuracy ≥ 0.65`.
   - **C**: `score > 0` (at least some tags ≥80% fields), then push it up.
3. Honest `docs/results_and_limitations_v3.md`: what worked, measured
   trajectory, remaining limits, and exact reproduction commands. Never
   overclaim; verify every number with the eval.

---

## 8. First moves

1. Scaffold `price_tag_v3` (copy v2's pipeline wiring; register a new name;
   confirm v2 unaffected).
2. Build the catalog resolver from `data/videos/*/*.csv` and wire it into a
   v2-style row pass first — measure the field-accuracy gain in isolation
   (this is fast and may alone move the needle a lot).
3. Build the close-pass clip harness.
4. Add track_crop_bank → multi_view_fusion → barcode-from-bars; measure each
   addition on the clip with the eval.
5. Iterate up the §7 ladder. Commit working milestones; keep v2 intact.

Be rigorous, measure everything with the provided eval, keep cycles fast,
and treat the catalog resolver + barcode-from-bars + multi-view fusion as the
three creative escapes from the resolution wall that defeated the single
-frame approach.
