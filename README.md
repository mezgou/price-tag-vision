# price-tag-vision

Minimal MVP for asynchronous price tag extraction from retail shelf videos

## What it is

This project is a local end-to-end wrapper around a future CV pipeline

Right now it lets you upload an MP4, create a processing job, wait for async completion, inspect a preview, view crop images, and download a CSV

## Current state

- full async stack (frontend / backend / worker / MinIO) is working
- **`price_tag_v2` is the real pipeline**: camera lens-undistort + 90° upright,
  YOLOv8m detector, **Ultralytics ByteTrack** (one stable track per tag),
  per-track top-K crops, lean QR cascade, **zonal OCR** (PaddleOCR-en for
  numeric zones + Tesseract-rus for the name), rubles+kopecks price assembly,
  bidirectional QR↔visible field backfill, per-template `нет` defaults,
  recognition-aware best-frame selection, exact bbox→raw 3840×2160 mapping,
  one deduplicated row per tag, 29-column CSV
- `price_tag_cpu_v1` and `mock` remain as fallback runtimes
- **Honest measured results, trajectory, and the resolution ceiling are in
  [docs/results_and_limitations.md](docs/results_and_limitations.md).** Short
  version: detection / tracking / dedup / geometry / price+discount extraction
  are verified correct; the official ≥80%-fields-per-tag score is bounded by
  ~5 sub-resolution fields (barcode / SKU / date / code / exact name) on the
  provided far-camera footage — a documented external data limit, not missing
  pipeline logic.

## Stack

- React + Vite + TypeScript + Tailwind in the frontend
- FastAPI in the backend
- PostgreSQL for job metadata
- Redis + RQ for async jobs
- Python worker for background processing
- FastAPI vision service for pipeline execution
- MinIO for input and output artifacts
- Docker Compose for local orchestration

## Architecture

User -> Frontend -> Backend API -> PostgreSQL + Redis/RQ -> Worker -> Vision Service -> MinIO -> Frontend result view

Backend owns job creation, status, and artifact access

Worker owns async execution

Vision service owns video processing and can later be replaced by a real CV/OCR pipeline

## Features available now

- upload a single MP4 from the web UI
- create a backend job with `POST /api/jobs`
- poll job status with `GET /api/jobs/{job_id}`
- show `status`, `progress`, `stage`, `message`, and `error`
- load preview with `GET /api/jobs/{job_id}/preview`
- load crops with `GET /api/jobs/{job_id}/crops`
- download CSV from `GET /api/jobs/{job_id}/download/csv`
- show a compact system status block in the frontend

## Known limitations (documented, not bugs)

- `barcode` / `id_sku` / `print_datetime` / `code` / exact Cyrillic
  `product_name` are sub-resolution on the provided far-camera 4K footage
  (~10 px text even on the closest crops) — see
  [docs/results_and_limitations.md](docs/results_and_limitations.md) §3–4
- PaddleOCR `lang='ru'` is broken in this paddle/paddleocr stack (`lang='en'`
  works and carries the numeric fields); paddle-GPU on Windows needs a system
  CUDA toolkit, so OCR runs on CPU (task allows this — speed is secondary)
- QR is ~2 % decodable at this camera distance (kept as a lean bonus; QR is
  non-critical per the task)
- SSE updates, auth, job-history dashboard not implemented

## Run locally

### Prerequisites

- Docker
- Docker Compose

### Setup

If `.env` does not exist yet, copy `.env.example` to `.env`

```sh
cp .env.example .env
```

Start the full stack

```sh
docker compose up --build
```

Open the app

- frontend: [http://localhost:5173](http://localhost:5173)
- backend: [http://localhost:8000](http://localhost:8000)
- backend status: [http://localhost:8000/api/system/status](http://localhost:8000/api/system/status)
- vision service: [http://localhost:9001](http://localhost:9001)
- MinIO console: [http://localhost:9002](http://localhost:9002)

MinIO default credentials

- username: `minioadmin`
- password: `minioadmin`

## Basic flow

1. Open the frontend at `http://localhost:5173`
2. Upload an `.mp4`
3. Wait until the job reaches `succeeded`
4. Review the preview table or JSON block
5. Review crop images
6. Download the CSV

## Useful commands

Stop the stack

```sh
docker compose down
```

Watch logs

```sh
docker compose logs -f backend worker vision-service frontend
```

### Run the v2 pipeline locally (no backend / MinIO)

Full run on a labelled video, then score against ground truth:

```sh
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py \
  --video data/videos/26_12-20/26_12-20.mp4 --pipeline price_tag_v2 \
  --output-dir artifacts/run --job-id r1 \
  --sample-fps 2 --max-frames 140 --max-total-crops 600 --max-crops-per-frame 40

uv run --project vision_service python vision_service/scripts/eval_price_tag_predictions.py \
  --pred artifacts/run/outputs/r1/result.csv --gt data/videos/26_12-20/26_12-20.csv
```

### Fast iteration harness

OCR on CPU is the bottleneck (~30–50 s/tag), so iterate on a **short
close-pass clip**, not the full video:

```sh
# one-time: cut a ~16 s clip around a close-up pass (frames 680–1010)
uv run --project vision_service python - <<'PY'
import cv2
c=cv2.VideoCapture("data/videos/26_12-20/26_12-20.mp4");fps=c.get(5)
w=int(c.get(3));h=int(c.get(4));o=cv2.VideoWriter("artifacts/clip_close.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
c.set(1,680)
[o.write(c.read()[1]) for _ in range(330)]
c.release();o.release();print("clip_close.mp4 written")
PY

uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py \
  --video artifacts/clip_close.mp4 --pipeline price_tag_v2 \
  --output-dir artifacts/q --job-id qc \
  --sample-fps 4 --max-frames 70 --max-total-crops 160 --max-crops-per-frame 30
```

Build / retrain the detector dataset (auto-labelled, no manual annotation):

```sh
uv run --project vision_service python vision_service/scripts/build_yolo_price_tag_dataset.py \
  --input-root data/videos --output-dir artifacts/datasets/price_tag_yolo_v2 \
  --undistort --propagate-frames 24 --propagation-ncc-threshold 0.42
```

## Repo layout

- `frontend/` web UI
- `backend/` API, job lifecycle, storage access, status checks
- `vision_service/` pipeline service
- `shared/` shared CSV schema

## Notes

`price_tag_v2` is the production pipeline. Engineering decisions, the measured
results trajectory, the precise resolution ceiling, and the path to a higher
score are documented honestly in
[docs/results_and_limitations.md](docs/results_and_limitations.md) and
[docs/executive_summary_ocr_pipeline.md](docs/executive_summary_ocr_pipeline.md).
Tested on Windows 11 + RTX 3060 Laptop 6 GB; YOLO/ByteTrack on GPU, QR/OCR on
CPU; ~30–50 s per tag.

It will need another pass once the real vision pipeline, README-grade setup details, and final demo flow are in place
