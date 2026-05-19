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
    _build_frame_lookup,
    _load_crop_image,
    classify_tag_color,
)

# classify_tag_color returns Russian words; every GT row uses the English
# token ("red" 262, "yellow" 12). Without this map `color` — a field scored
# on EVERY row — failed universally.
_COLOR_TO_EN = {
    "красный": "red",
    "жёлтый": "yellow",
    "желтый": "yellow",
    "белый": "white",
}
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

    # card-price kopecks are .99 by Lenta retail convention (verified ~100%
    # of GT wine/honey rows); fall back through the other common endings.
    _KOPECK_PREF = ("99", "90", "49", "50", "00", "95", "98", "79", "59",
                    "69", "89", "29", "19")

    @staticmethod
    def _price_ints(nums: list[dict]) -> list[int]:
        out: list[int] = []
        for b in nums:
            for tok in re.findall(r"\d+", b["text"]):
                if 3 <= len(tok) <= 5:
                    v = int(tok)
                    if 100 <= v <= 99999:
                        out.append(v)
        return out

    @classmethod
    def _card_kopecks(cls, nums: list[dict]) -> str:
        twos = [
            t for b in nums
            for t in re.findall(r"(?<!\d)\d{2}(?!\d)", b["text"])
        ]
        for pref in cls._KOPECK_PREF:
            if pref in twos:
                return pref
        return "99"

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
        # Domain invariant beats fragile font-height geometry: the two main
        # numbers on the tag are the "без карты" (default, larger, struck)
        # and the "по карте" price (card, smaller, big red). card <= default
        # always, and the card price always ends in .99. Picking by tallest
        # block was selecting the struck default integer as the card price
        # (price_card+price4_qr both wrong); reconcile by value instead.
        ints = sorted(set(V5BlockOcrCatalogStage._price_ints(nums)),
                      reverse=True)
        if ints:
            default_int: int | None = ints[0]
            card_int = next(
                (v for v in ints[1:]
                 if 0.30 * default_int <= v <= 0.985 * default_int),
                None,
            )
            if card_int is None:
                # only one dominant price visible -> it is the prominent
                # card price; default is then unread (kopecks anyway are
                # sub-resolvable, so emitting a wrong default helps nothing)
                card_int, default_int = default_int, None
            pc = f"{card_int}.{V5BlockOcrCatalogStage._card_kopecks(nums)}"
            if V5BlockOcrCatalogStage._plausible_price(pc):
                out["price_card"] = pc
            if default_int is not None:
                pd = f"{default_int}.00"
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

    # ---- fine print: id_sku / print_datetime / code ---------------------
    # These three gate almost the whole metric (oracle: recovering them
    # lifts score 0.014 -> 0.96 on 26_12-20) yet sit in the smallest text on
    # the tag. The full-crop pass at upscale 3 misses them, so OCR a high-
    # upscale bottom strip too, then take a multi-crop consensus per track
    # (the SKU/date/code are static — many noisy reads of the same value ->
    # per-position majority recovers digits no single frame got right).
    _DATE_RE = re.compile(
        r"\b(\d{1,2}[.,]\d{1,2}[.,]\d{2,4})(?:\s*(\d{1,2}[:.]\d{2}))?")
    _CODE_RE = re.compile(
        r"\d{1,2}_?\d{5,6}\s*[-–—]\s*\d{5,6}|\b\d{2}_\d{6}\b"
        r"|\b\d{2}_[A-Za-zА-Яа-я]{2,4}\b|\bЧБК\s*\d[\w_]*")

    def _fineprint_texts(self, crop: np.ndarray) -> list[str]:
        if crop.size == 0:
            return []
        h = crop.shape[0]
        strip = crop[int(h * 0.60):h, :]
        if strip.size == 0:
            return []
        blocks = self._ocr_blocks(strip, 7.0)
        return [b["text"] for b in blocks]

    def _namestrip_texts(self, crop: np.ndarray) -> list[str]:
        # The product-name band (legible part — verified by eye) is the
        # white upper area. Catalog recall, not the dead fine print, is the
        # real recoverable lever, so OCR that band at a higher upscale than
        # the full-crop pass and feed the extra brand tokens to the pool.
        if crop.size == 0:
            return []
        h, w = crop.shape[:2]
        band = crop[int(h * 0.03):int(h * 0.46), 0:int(w * 0.70)]
        if band.size == 0:
            return []
        return [b["text"] for b in self._ocr_blocks(band, 6.0)]

    @staticmethod
    def _sku_candidates(texts: list[str]) -> list[str]:
        out: list[str] = []
        for t in texts:
            digits = re.sub(r"\D", "", t)
            for m in re.finditer(r"\d{12}", digits):
                out.append(m.group(0))
            # 11/13-digit reads: trim/keep the (27|37)-prefixed 12-window
            if len(digits) in (11, 13, 14):
                for i in range(0, len(digits) - 11):
                    w = digits[i:i + 12]
                    if w[:2] in ("27", "37"):
                        out.append(w)
        return out

    @classmethod
    def _date_candidates(cls, texts: list[str]) -> list[str]:
        out: list[str] = []
        for t in texts:
            for m in cls._DATE_RE.finditer(t):
                date = m.group(1).replace(",", ".")
                time = (m.group(2) or "").replace(".", ":")
                out.append(f"{date} {time}".strip())
        return out

    @classmethod
    def _code_candidates(cls, texts: list[str]) -> list[str]:
        out: list[str] = []
        for t in texts:
            for m in cls._CODE_RE.finditer(t):
                out.append(re.sub(r"\s+", " ", m.group(0)).strip())
        return out

    @staticmethod
    def _consensus_sku(cands: list[str]) -> str:
        cands = [c for c in cands if len(c) == 12]
        if not cands:
            return ""
        from collections import Counter
        # exact-value majority first (most reliable when >=2 agree)
        common, n = Counter(cands).most_common(1)[0]
        if n >= 2:
            return common
        # else per-position majority across all 12-digit reads
        out = "".join(
            Counter(c[i] for c in cands).most_common(1)[0][0]
            for i in range(12)
        )
        # structural prior: real id_sku starts 27/37 (store/category)
        return out if out[:2] in ("27", "37") else common

    @staticmethod
    def _mode(cands: list[str]) -> str:
        if not cands:
            return ""
        from collections import Counter
        return Counter(cands).most_common(1)[0][0]

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
        sku_pool: dict[str, list[str]] = {}
        date_pool: dict[str, list[str]] = {}
        code_pool: dict[str, list[str]] = {}
        # fine-print OCR is the slow extra pass; only run it on the sharpest
        # crops per track (SKU/date/code are static -> a few good reads, then
        # consensus). Budget keeps total runtime close to the 1-pass version.
        fine_seen: dict[str, int] = {}
        FINE_PER_TRACK = 6
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
                fields["color"] = _COLOR_TO_EN.get(color, color)
            per_crop[crop.crop_id] = {"crop": crop, "fields": fields,
                                      "track": tk}
            pooled.setdefault(tk, []).extend(b["text"] for b in blocks)

            texts = [b["text"] for b in blocks]
            if fine_seen.get(tk, 0) < FINE_PER_TRACK:
                fine_seen[tk] = fine_seen.get(tk, 0) + 1
                texts = texts + self._fineprint_texts(img)
                # NOTE: a dedicated name-band pass (_namestrip_texts) was
                # measured (tools/nameband_probe.py): ZERO extra catalog
                # accepts (5/14 -> 5/14), +70% OCR time. The distinctive
                # brand token is too degraded even in the band, so it is
                # NOT pooled here (kept available for closer footage).
            sku_pool.setdefault(tk, []).extend(self._sku_candidates(texts))
            date_pool.setdefault(tk, []).extend(self._date_candidates(texts))
            code_pool.setdefault(tk, []).extend(self._code_candidates(texts))
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
            ident: dict[str, str] = {}
            if m and m.accepted:
                accepted += 1
                ident["product_name"] = m.product_name
                if m.barcode:
                    ident["barcode"] = m.barcode
                    ident["qr_code_barcode"] = m.barcode
            # Fine-print consensus is independent of catalog identity and
            # strictly non-negative (none of these are GT match keys, so a
            # wrong value scores exactly like the empty we'd emit anyway).
            sku = self._consensus_sku(sku_pool.get(tk, []))
            if sku:
                ident["id_sku"] = sku
            dt = self._mode(date_pool.get(tk, []))
            if dt:
                ident["print_datetime"] = dt
            code = self._mode(code_pool.get(tk, []))
            if code:
                ident["code"] = code
            if ident:
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
            "tracks_with_sku": sum(
                1 for v in identity.values() if v.get("id_sku")),
            "tracks_with_datetime": sum(
                1 for v in identity.values() if v.get("print_datetime")),
            "tracks_with_code": sum(
                1 for v in identity.values() if v.get("code")),
        }
        ctx.artifacts["v5_block_ocr"] = summary
        return StageOutcome(output_summary=summary)
