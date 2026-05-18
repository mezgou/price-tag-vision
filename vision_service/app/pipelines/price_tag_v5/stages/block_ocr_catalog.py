"""v5 identity stage — validated logic, integrated.

Per top-K crop: ONE PaddleOCR-en full-crop pass (fast, ~1s, no timeout).
Parse price_card/price_default/discount/special/color by block geometry
(measured-good). Pool brand tokens per ByteTrack track -> CatalogResolver
-> exact product_name (+ safe barcode). Write fields into
crop.attributes['ocr']['fields'] so the proven v2 RowFusionStage does
QR<->visible backfill, нет-defaults, best-frame and dedup unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, StageOutcome
from app.pipelines.price_tag_v2.stages.zonal_ocr import (
    _best_discount,
    _best_price,
    _build_frame_lookup,
    _load_crop_image,
    classify_tag_color,
)
from app.pipelines.price_tag_v5.catalog import CatalogResolver

import cv2  # noqa: E402


@dataclass(slots=True)
class _Cfg:
    enabled: bool
    max_crops: int
    upscale: float
    category: str
    db_path: str

    @classmethod
    def from_context(cls, ctx: PipelineContext) -> "_Cfg":
        raw = ctx.config.get("v5_block_ocr", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            max_crops=int(raw.get("max_crops", 160)),
            upscale=float(raw.get("upscale", 3.0)),
            category=str(raw.get("category", "wine")),
            db_path=str(raw.get("db_path", "data/db_hack.csv")),
        )


class V5BlockOcrCatalogStage(BaseStage):
    name = "V5BlockOcrCatalogStage"

    def __init__(self) -> None:
        self._engine: Any = None
        self._resolver: CatalogResolver | None = None

    def describe_input(self, ctx: PipelineContext) -> dict[str, Any]:
        return {"crops": len(ctx.crop_candidates),
                "detections": len(ctx.detections)}

    def _get_engine(self) -> Any:
        if self._engine is None:
            import torch  # noqa: F401  # before paddleocr (albumentations DLL)
            from paddleocr import PaddleOCR
            for kw in ({"lang": "en", "use_textline_orientation": True},
                       {"lang": "en", "use_angle_cls": True},
                       {"lang": "en"}):
                try:
                    self._engine = PaddleOCR(**kw)
                    break
                except Exception:  # noqa: BLE001
                    continue
        return self._engine

    def _ocr_blocks(self, crop: np.ndarray, up: float) -> list[dict]:
        eng = self._get_engine()
        if eng is None or crop.size == 0:
            return []
        img = cv2.resize(crop, None, fx=up, fy=up,
                         interpolation=cv2.INTER_CUBIC)
        H = img.shape[0]
        try:
            res = eng.ocr(img) if not hasattr(eng, "predict") else eng.predict(img)
        except Exception:  # noqa: BLE001
            return []
        blocks: list[dict] = []
        page = res[0] if isinstance(res, list) and res else res
        if isinstance(page, dict):  # 3.x predict
            tx = page.get("rec_texts", []) or []
            sc = page.get("rec_scores", []) or []
            pl = page.get("rec_polys") or page.get("dt_polys") or []
            items = [
                (np.array(pl[i]), tx[i], float(sc[i]) if i < len(sc) else 0.0)
                for i in range(len(tx)) if i < len(pl)
            ]
        else:  # legacy .ocr()
            items = []
            for it in page or []:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    pay = it[1]
                    if isinstance(pay, (list, tuple)) and len(pay) >= 2:
                        items.append((np.array(it[0]), str(pay[0]),
                                      float(pay[1])))
        for poly, txt, conf in items:
            if poly is None or poly.size == 0 or not str(txt).strip():
                continue
            ys, xs = poly[:, 1], poly[:, 0]
            rel_h = float(ys.max() - ys.min()) / float(H)
            rel_y = float(ys.min()) / float(H)
            blocks.append({"text": str(txt), "conf": conf,
                           "rel_h": rel_h, "rel_y": rel_y})
        return blocks

    @staticmethod
    def _plausible_price(v: str) -> bool:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        return 10.0 <= f <= 100000.0  # rejects volume "0.75", junk

    @staticmethod
    def _prices_from_blocks(blocks: list[dict]) -> dict[str, str]:
        # drop volume tokens (0.75 L / 1.5L / 750 ml) before price parsing
        nums = [
            b for b in blocks
            if re.search(r"\d", b["text"]) and "%" not in b["text"]
            and not re.search(r"\d[\s.,]*(L|Л|ML|МЛ)\b", b["text"].upper())
            and not re.fullmatch(r"\s*[01][.,]?\d{1,2}\s*[lLлЛ]?\s*", b["text"])
        ]
        out: dict[str, str] = {}
        if nums:
            card_b = max(nums, key=lambda b: b["rel_h"])
            near = " ".join(
                b["text"] for b in nums
                if abs(b["rel_y"] - card_b["rel_y"]) < 0.18
            )
            pc = _best_price(near or card_b["text"])
            if V5BlockOcrCatalogStage._plausible_price(pc):
                out["price_card"] = pc
            up_txt = " ".join(
                b["text"] for b in nums
                if b["rel_y"] < 0.45 and b is not card_b
            )
            pd = _best_price(up_txt)
            if (V5BlockOcrCatalogStage._plausible_price(pd)
                    and pd != out.get("price_card")):
                out["price_default"] = pd
        disc = _best_discount(" ".join(b["text"] for b in blocks))
        if disc:
            out["discount_amount"] = disc
        for b in blocks:
            t = b["text"].strip().upper()
            if t in ("К", "K", "Ш", "Л") and b["rel_y"] > 0.7:
                out["special_symbols"] = "К" if t in ("К", "K") else t
        return out

    def run(self, ctx: PipelineContext) -> StageOutcome:
        cfg = _Cfg.from_context(ctx)
        if not cfg.enabled or not ctx.crop_candidates:
            return StageOutcome(output_summary={"enabled": cfg.enabled,
                                                "crops": 0})
        if self._resolver is None:
            self._resolver = CatalogResolver(cfg.db_path)

        det2track = {
            d.detection_id: str(d.attributes.get("track_id") or d.detection_id)
            for d in ctx.detections
        }
        frame_lookup = _build_frame_lookup(ctx.sampled_frames)
        crops = sorted(ctx.crop_candidates,
                       key=lambda c: c.quality.score, reverse=True)[:cfg.max_crops]

        per_crop: dict[str, dict] = {}
        pooled: dict[str, list[str]] = {}
        hint: dict[str, str] = {}
        for crop in crops:
            img = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
            if img is None:
                continue
            blocks = self._ocr_blocks(img, cfg.upscale)
            if not blocks:
                continue
            tk = det2track.get(crop.detection_id, crop.detection_id)
            fields = self._prices_from_blocks(blocks)
            color = classify_tag_color(img)
            if color:
                fields["color"] = color
            per_crop[crop.crop_id] = {"crop": crop, "fields": fields,
                                      "track": tk}
            pooled.setdefault(tk, []).extend(b["text"] for b in blocks)
            if tk not in hint:
                for b in blocks:
                    d = re.sub(r"\D", "", b["text"])
                    if len(d) in (12, 13):
                        hint[tk] = d
                        break

        # one catalog resolve per track (exact name + safe barcode)
        identity: dict[str, dict] = {}
        accepted = 0
        for tk, texts in pooled.items():
            m = self._resolver.resolve(texts, category=cfg.category,
                                       barcode_hint=hint.get(tk, ""))
            if m and m.accepted:
                accepted += 1
                ident = {"product_name": m.product_name}
                if m.barcode:
                    ident["barcode"] = m.barcode
                    ident["qr_code_barcode"] = m.barcode
                identity[tk] = ident

        crops_with_fields = 0
        for cid, rec in per_crop.items():
            fields = dict(rec["fields"])
            fields.update(identity.get(rec["track"], {}))
            if fields:
                crops_with_fields += 1
            rec["crop"].attributes["ocr"] = {
                "text": "", "confidence": 0.0,
                "engine": "v5_block_paddle_en", "fields": fields,
            }
        summary = {
            "enabled": True,
            "crops_ocred": len(per_crop),
            "tracks": len(pooled),
            "catalog_accepted_tracks": accepted,
            "crops_with_fields": crops_with_fields,
        }
        ctx.artifacts["v5_block_ocr"] = summary
        return StageOutcome(output_summary=summary)
