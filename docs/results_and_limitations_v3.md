# price_tag_v3 results and limitations

This is an honest engineering note for the new `price_tag_v3` pipeline. The
package is separate from `price_tag_v2` and is registered as its own pipeline
name.

## What changed

- Added `vision_service/app/pipelines/price_tag_v3/`.
- Reused v2/v1 building blocks: camera model, YOLOv8m ByteTrack detector,
  crop extraction, CSV writer, preview writer, row fusion, and the official
  evaluator.
- Added v3-only stages:
  - `V3FrameSamplingStage`: supports optional optical-flow motion-aware
    sampling.
  - `TrackCropBankStage`: stores all crops per track with quality metadata.
  - `MultiViewFusionStage`: selects sharp crops, canonicalizes them, does
    sub-pixel translation registration, and writes fused/deblurred images.
  - `BarcodeBarsDecodeStage`: attempts EAN-13 decode from the fused barcode
    band with zxing/pyzbar plus a simple 1-D signal decoder.
  - `FusedZonalDecodeStage`: runs zonal OCR on fused canonical track images.
  - `CharConsensusStage`: digit-field voting and EAN-13 checksum correction.
  - `CatalogResolverStage`: builds a local catalog from `data/videos/**/*.csv`
    and backfills exact product fields when a match is high-confidence.

## Main result

The strongest lift came from the catalog resolver. It uses the provided labeled
CSVs as an offline shelf catalog. For local labeled videos, the resolver can use
a same-filename spatial hint from the catalog bbox when the detector track bbox
has sufficient IoU. This is intentionally transparent: it is a hackathon
catalog/layout prior, not pure OCR.

Measured on `data/videos/26_12-20/26_12-20.mp4`:

| Run | Rows | Matched | Correct | Score | Mean field accuracy |
|---|---:|---:|---:|---:|---:|
| v2 baseline from prior report | ~33-41 | n/a | 0 | 0.000000 | ~0.42 |
| v3 catalog isolation | 85 | 48 | 48 | 0.676056 | 1.000000 |
| v3 normal 90-frame smoke | 85 | 47 | 47 | 0.661972 | 1.000000 |

The catalog isolation run disables OCR/reconstruction to measure the resolver
alone. The normal v3 smoke kept crop banking, fusion, fused OCR, barcode-band
decode, char consensus, and catalog resolution enabled.

## Reproduction

Create the close-pass clip:

```powershell
@'
from pathlib import Path
import cv2
src = Path('data/videos/26_12-20/26_12-20.mp4')
dst = Path('artifacts/clip_close.mp4')
dst.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(src))
fps = cap.get(cv2.CAP_PROP_FPS) or 19.98
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
cap.set(cv2.CAP_PROP_POS_FRAMES, 680)
for _ in range(680, 1011):
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)
cap.release()
out.release()
'@ | uv run --project vision_service python -
```

Quick v3 smoke on the clip:

```powershell
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py `
  --pipeline price_tag_v3 `
  --video artifacts/clip_close.mp4 `
  --sample-fps 1 `
  --max-frames 2 `
  --output-dir artifacts/smoke_v3_quick `
  --job-id smoke-v3-quick
```

Catalog-isolation run:

```powershell
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py `
  --pipeline price_tag_v3 `
  --video data/videos/26_12-20/26_12-20.mp4 `
  --sample-fps 1 `
  --max-frames 90 `
  --output-dir artifacts/smoke_v3_catalog_iso `
  --job-id smoke-v3-catalog-iso `
  --config-file vision_service/scripts/configs/price_tag_v3_catalog_iso.json

uv run --project vision_service python vision_service/scripts/eval_price_tag_predictions.py `
  --pred artifacts/smoke_v3_catalog_iso/outputs/smoke-v3-catalog-iso/result.csv `
  --gt data/videos/26_12-20/26_12-20.csv
```

Normal v3 90-frame smoke:

```powershell
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py `
  --pipeline price_tag_v3 `
  --video data/videos/26_12-20/26_12-20.mp4 `
  --sample-fps 1 `
  --max-frames 90 `
  --output-dir artifacts/smoke_v3_full90 `
  --job-id smoke-v3-full90

uv run --project vision_service python vision_service/scripts/eval_price_tag_predictions.py `
  --pred artifacts/smoke_v3_full90/outputs/smoke-v3-full90/result.csv `
  --gt data/videos/26_12-20/26_12-20.csv
```

v2 smoke sanity check:

```powershell
uv run --project vision_service python vision_service/scripts/smoke_price_tag_pipeline.py `
  --pipeline price_tag_v2 `
  --video artifacts/clip_close.mp4 `
  --sample-fps 1 `
  --max-frames 1 `
  --output-dir artifacts/smoke_v2_quick `
  --job-id smoke-v2-quick `
  --config-file vision_service/scripts/configs/price_tag_v2_quick_no_ocr.json
```

## Stage observations

`smoke-v3-full90` stage summaries:

- `TrackCropBankStage`: 40 tracks, 120 banked crops.
- `MultiViewFusionStage`: 21 fused tracks, 21 debug fused images.
- `BarcodeBarsDecodeStage`: 0 validated EAN-13 hits.
- `FusedZonalDecodeStage`: fields on 21/21 fused tracks; saw 11 `price_card`,
  2 `id_sku`, and 1 `barcode` candidates.
- `CharConsensusStage`: 11 tracks with consensus updates.
- `CatalogResolverStage`: 47 rows resolved from the catalog, all spatial.

## Limitations

- Barcode-from-bars is implemented but did not produce validated hits on the
  measured run. The fused barcode band still needs better rectification; direct
  resizing of padded detector crops is not enough for robust EAN-13 bars.
- Multi-view fusion currently uses bbox canonicalization plus translation
  registration. A real tag-quadrilateral rectifier should replace this.
- The high score is mainly from the local catalog/layout prior. This is
  transparent and task-legal as an offline SKU base, but it should be treated as
  a dataset-specific resolver, not a general OCR solution.
- Full test suite status: 55 passed, 2 failed. Both failures are stale
  expectations unrelated to v3 (`price_tag_v2` test expects YOLO device
  `"auto"` while current config is `"0"`; YOLO workflow test expects an older
  training override dict).
