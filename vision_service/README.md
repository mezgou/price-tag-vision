# vision-service

`vision_service` hosts the local runtime pipelines behind `POST /api/pipeline/process`.
`price_tag_cpu_v1` remains the stable legacy fallback. `price_tag_v2` adds the
YOLO/QR-first MVP path from the OCR architecture brief.

## v2 runtime stages

1. `FrameMetadataStage`
2. `FrameSamplingStage`
3. `YoloDetectionStage`
4. `FallbackHeuristicCandidateDetectionStage`
5. `CropExtractionStage`
6. `BarcodeQrDecodeStage`
7. `RowFusionStage`
8. `CsvWriterStage`
9. `PreviewWriterStage`
10. `DebugManifestStage`

`YoloDetectionStage` is optional at runtime. If Ultralytics or trained weights are
not available, v2 falls back to the existing color/geometry detector and still
writes CSV, preview, and manifest artifacts.

## Legacy runtime stages

1. `FrameMetadataStage`
2. `FrameSamplingStage`
3. `HeuristicCandidateDetectionStage`
4. `CropExtractionStage`
5. `BarcodeQrDecodeStage`
6. `CsvWriterStage`
7. `PreviewWriterStage`
8. `DebugManifestStage`

The runtime stays brand-neutral and keeps the API contract unchanged:

- `POST /api/pipeline/process`
- response fields: `csv_key`, `preview_key`, `crop_keys`, `stats`
- `result.csv` follows `shared/csv_schema.py`

## Direct local smoke

Run the pipeline directly on a local MP4 without backend or MinIO:

```bash
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py \
  --video data/videos/43_15/43_15.mp4 \
  --output-dir artifacts/smoke/43_15 \
  --pipeline price_tag_v2 \
  --max-frames 20
```

The script writes artifacts under:

```text
artifacts/smoke/43_15/outputs/{job_id}/...
```

Useful outputs:

- `preview.json`
- `debug/pipeline_manifest.json`
- `debug/frames/*.jpg`
- `debug/overlays/*.jpg`
- `debug/masks/*.jpg`
- `debug/crops/*.jpg`
- `debug/contact_sheets/top_crops.jpg`
- `smoke_summary.json`

## Debug artifacts

The current tuning pass focuses on better diagnostics before OCR:

- explainable detection attributes per bbox
- white/accent/combined masks
- crop quality summary
- top-crops contact sheet
- preview and manifest counts that stay consistent with saved debug keys

## YOLO dataset build

Build a local YOLO-format dataset from official labeled video/CSV pairs:

```bash
uv run --project vision_service python vision_service/scripts/build_yolo_price_tag_dataset.py \
  --input-root data/videos \
  --output-dir artifacts/datasets/price_tag_yolo_v1 \
  --orientation-mode none \
  --propagate-frames 20 \
  --propagation-ncc-threshold 0.42 \
  --make-orientation-qa
```

The builder reads official CSV bboxes in raw video coordinates. With
`--propagate-frames 20`, each train-split bbox is template-matched onto
neighboring frames by NCC, producing extra YOLO labels without manual clicks.

Expected outputs:

```text
artifacts/datasets/price_tag_yolo_v1/
  dataset.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  qa/overlays/*.jpg
  qa/orientation_check/{mode}/*.jpg
  qa/contact_sheets/*.jpg
  qa/dataset_report.json
```

## YOLO training scaffold

The runtime does not depend on Ultralytics. Training and export stay in dev-only wrappers.

Preferred training command:

```bash
uv run --with ultralytics python vision_service/scripts/train_yolo_price_tag_detector.py \
  --data artifacts/datasets/price_tag_yolo_v1/dataset.yaml \
  --model yolo11n.pt \
  --imgsz 1280 \
  --epochs 120 \
  --batch 2 \
  --device 0 \
  --project artifacts/models \
  --name price_tag_yolo11n_v1 \
  --patience 25 \
  --seed 42
```

If the current environment does not expose CUDA to PyTorch, use the CPU fallback:

```bash
uv run --with ultralytics python vision_service/scripts/train_yolo_price_tag_detector.py \
  --data artifacts/datasets/price_tag_yolo_v1/dataset.yaml \
  --model yolo11n.pt \
  --imgsz 960 \
  --epochs 30 \
  --batch 1 \
  --device 0 \
  --project artifacts/models \
  --name price_tag_yolo11n_v1 \
  --patience 10 \
  --seed 42 \
  --allow-cpu-fallback
```

Generate qualitative QA overlays after training:

```bash
uv run --with ultralytics python vision_service/scripts/predict_yolo_price_tag_qa.py \
  --model artifacts/models/price_tag_yolo11n_v1/weights/best.pt \
  --data artifacts/datasets/price_tag_yolo_v1/dataset.yaml \
  --imgsz 1280 \
  --device 0
```

## Local CSV evaluation

Evaluate a prediction CSV against local labeled CSV files:

```bash
uv run --project vision_service python vision_service/scripts/eval_price_tag_predictions.py \
  --pred artifacts/smoke/43_15/outputs/smoke-43_15/result.csv \
  --gt data/videos \
  --iou 0.5 \
  --field-accuracy 0.8
```

Root-level wrappers are also available for the architecture brief commands:

```bash
uv run --project vision_service python tools/build_yolo_dataset.py --input-root data/videos
uv run --project vision_service python tools/eval_local.py --pred result.csv --gt data/videos
```
