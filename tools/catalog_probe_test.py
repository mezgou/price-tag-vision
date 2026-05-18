"""Fast standalone test of CatalogResolver on real probe OCR.

No pipeline, no GPU. Uses artifacts/ocr_probe/report.json (real garbled OCR
blocks from the 26_12-20 close clip) and measures precision vs 26_12-20 GT
barcodes/names. Confirms (or refutes) the core v5 hypothesis instantly.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
from pathlib import Path

R = Path.cwd()
sys.path.insert(0, str(R / "vision_service"))
sys.path.insert(1, str(R))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.pipelines.price_tag_v5.catalog import CatalogResolver

rep = json.loads((R / "artifacts/ocr_probe/report.json").read_text("utf-8"))

# GT for the source video of the clip (frames 680-1010 of 26_12-20)
gt_bc: set[str] = set()
gt_names: list[str] = []
for row in csv.DictReader(open(R / "data/videos/26_12-20/26_12-20.csv", encoding="utf-8")):
    b = re.sub(r"\D", "", row.get("barcode") or "")
    if len(b) in (12, 13):
        gt_bc.add(b)
    nm = (row.get("product_name") or "").strip()
    if nm and nm.lower() != "нет":
        gt_names.append(nm)

t0 = time.time()
res = CatalogResolver()
print(f"catalog loaded in {time.time()-t0:.1f}s | "
      f"GT barcodes={len(gt_bc)} GT names={len(set(gt_names))}")

acc = bc_in_gt = name_hit = 0
for r in rep:
    texts = [b["text"] for b in r["blocks"]]
    digits = [b["text"] for b in r["blocks"]
              if re.fullmatch(r"\d{12,13}", re.sub(r"\D", "", b["text"]))]
    hint = re.sub(r"\D", "", digits[0]) if digits else ""
    m = res.resolve(texts, category="wine", barcode_hint=hint)
    if m is None:
        print(f"trk{r['track']:>3} {r['crop']:<9} NO-BRAND")
        continue
    in_gt = m.barcode in gt_bc
    nmatch = any(
        m.product_name[:20].casefold() in g.casefold()
        or g[:20].casefold() in m.product_name.casefold()
        for g in gt_names
    )
    acc += m.accepted
    bc_in_gt += m.accepted and in_gt
    name_hit += m.accepted and nmatch
    flag = "OK " if (m.accepted and in_gt) else ("acc" if m.accepted else "—  ")
    print(f"trk{r['track']:>3} {r['crop']:<9} {flag} "
          f"s={m.score:>3.0f} mg={m.margin:>4.1f} bc={m.barcode or '-':<14}"
          f"{'∈GT' if in_gt else '   '} {m.product_name[:46]}")

n = len(rep)
print(f"\naccepted={acc}/{n}  barcode∈GT(of accepted)={bc_in_gt}  "
      f"name∈GT(of accepted)={name_hit}")
print("PRECISION of accepted-barcode = "
      f"{(bc_in_gt/acc if acc else 0):.0%}  (wrong barcode mis-keys GT)")
