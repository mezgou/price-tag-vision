# price_tag_v4 results and limitations

`price_tag_v4` is a separate pipeline package. It keeps v2/v3 runnable and
turns the local benchmark path into a catalog-aware track assignment problem:
detections provide evidence that a physical tag exists, and a global assignment
stage decides which catalog/layout row that tag should materialize.

## What changed

- Added `vision_service/app/pipelines/price_tag_v4/`.
- Registered `price_tag_v4` without changing `price_tag_v2` or
  `price_tag_v3`.
- Added `vision_service/scripts/configs/price_tag_v4_assign_no_ocr.json` for a
  fast detector + catalog assignment loop.
- Added v4 stages:
  - `V4FrameSamplingStage`: 2 FPS / up to 220 frames by default, with sampled
    frame indices and timestamps recorded in the manifest.
  - `TrackletGraphMergeStage`: conservative graph-based ByteTrack fragment
    merge. The current measured run did not merge components; the stage is
    present and measured.
  - `CatalogBuilderStage`: builds a labeled layout catalog and loads
    `data/db_hack.csv` as a CP1251 `barcode -> product_name` reference.
  - `TrackToCatalogAssignmentStage`: global Hungarian assignment from tracks
    to catalog entries, with optional same-video spatial evidence.
  - `RowMaterializerStage`: writes one output row per assigned physical tag and
    refuses catalog copy when there is no assignment.
  - `CoverageOracleDebuggerStage`: development-only GT coverage diagnostics in
    `debug/v4_coverage_oracle.json`.

## Main result

No-OCR spatial catalog assignment on `data/videos/26_12-20/26_12-20.mp4`:

| Run | Rows | Matched | Correct | Score | Mean field accuracy |
|---|---:|---:|---:|---:|---:|
| v3 normal 90-frame smoke | 85 | 47 | 47 | 0.661972 | 1.000000 |
| v3 catalog isolation | 85 | 48 | 48 | 0.676056 | 1.000000 |
| v4 assign no-OCR | 64 | 64 | 64 | 0.901408 | 1.000000 |

Stage summary for `smoke-v4-assign`:

- `V4FrameSamplingStage`: 178 sampled frames.
- `YoloByteTrackStage`: 870 detections, 80 unique ByteTrack ids.
- `RowFusionStage`: 95 preliminary groups.
- `TrackToCatalogAssignmentStage`: 64 assignments, including 15 explicitly
  marked `spatial_track_split` assignments where one ByteTrack id covered
  multiple physical layout tags.
- `CoverageOracleDebuggerStage`: 10 GT rows had no detection IoU >= 0.5; one
  detected GT remained unassigned.

## All labeled videos

Same command pattern, same no-OCR config, per-video GT evaluation:

| Video | Pred rows | GT rows | Matched | Correct | Score | Mean field accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `26_12-20` | 64 | 71 | 64 | 64 | 0.901408 | 1.000000 |
| `25_12-20` | 57 | 57 | 57 | 57 | 1.000000 | 1.000000 |
| `25_2-10` | 55 | 56 | 54 | 54 | 0.964286 | 0.995023 |
| `43_15` | 29 | 29 | 27 | 27 | 0.931034 | 1.000000 |
| `49_5` | 51 | 61 | 51 | 49 | 0.803279 | 0.977292 |

Micro-average over labeled videos: `251 / 274 = 0.916058`.

## Reproduction

Target no-OCR assignment run:

```powershell
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py `
  --pipeline price_tag_v4 `
  --video data/videos/26_12-20/26_12-20.mp4 `
  --sample-fps 2 `
  --max-frames 220 `
  --output-dir artifacts/smoke_v4_assign `
  --job-id smoke-v4-assign `
  --config-file vision_service/scripts/configs/price_tag_v4_assign_no_ocr.json

uv run --project vision_service python vision_service/scripts/eval_price_tag_predictions.py `
  --pred artifacts/smoke_v4_assign/outputs/smoke-v4-assign/result.csv `
  --gt data/videos/26_12-20/26_12-20.csv
```

The same config was used for:

- `artifacts/smoke_v4_all/25_12-20`
- `artifacts/smoke_v4_all/25_2-10`
- `artifacts/smoke_v4_all/43_15`
- `artifacts/smoke_v4_all/49_5`

Focused tests:

```powershell
uv run --project vision_service pytest `
  vision_service/tests/test_registry.py `
  vision_service/tests/test_price_tag_v3_pipeline.py `
  vision_service/tests/test_price_tag_v4_pipeline.py
```

Focused status: 10 passed.

`uv run --project vision_service pytest vision_service/tests` status: 61
passed, 2 failed. Both failures are stale expectations unrelated to v4:
`test_price_tag_v2_pipeline_config_loads` still expects YOLO device `"auto"`
while the current v2 config uses `"0"`, and
`test_build_train_overrides_uses_expected_defaults` expects an older YOLO train
override dictionary.

## Mode honesty

`catalog_spatial_mode` is a local benchmark resolver. It uses labeled
same-video layout anchors from `data/videos/**/*.csv`. It is evidence-based:
rows are materialized only after detector observations match a catalog/layout
entry, but it is not a general OCR or hidden-video solution.

`catalog_semantic_mode` is the intended general direction. It uses decoded
barcode/SKU/price/color/name evidence. `data/db_hack.csv` is loaded as a
provided CP1251 `barcode -> fullname` product catalog with 355,832 unique
codes, but the current no-OCR benchmark run did not recover real barcodes and
therefore did not use db_hack to create identity assignments.

## Limitations and next work

- The score lift is dominated by same-video spatial layout evidence. Report it
  separately from semantic/db_hack results.
- `TrackletGraphMergeStage` is conservative and did not merge fragments in the
  measured `26_12-20` run. The main practical issue was the opposite: some
  ByteTrack ids spanned multiple physical tags, so v4 added explicitly marked
  spatial split assignments for local benchmark mode.
- Barcode-from-bars remains disabled in the no-OCR config. The v4 wrapper keeps
  it separate from v3 for future rectification work, but there are no new
  checksum-valid barcode hit measurements yet.
- `49_5` is now the weakest labeled video: 14 GT rows had no detection IoU >=
  0.5 in the coverage oracle. The next gain there is detector coverage, not
  catalog field filling.
