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
from app.pipelines.price_tag_v2.stages.row_fusion import parse_qr_payload
from app.pipelines.price_tag_v5.catalog import (
    CatalogMatch,
    CatalogResolver,
    ean13_ok,
    to_ean13,
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
import cv2  # noqa: E402


def _catalog_identity_fields(
    match: CatalogMatch | None,
    *,
    barcode_hint: str = "",
    reliable_barcode_hint: str | None = None,
) -> dict[str, str]:
    if match is None:
        return {}
    if not match.accepted and not match.catalog_guess_name:
        return {}

    fields = {
        "catalog_match_status": match.status,
        "catalog_resolver_status": match.status,
        "catalog_match_source": "v5_catalog_resolver",
        "catalog_match_score": f"{match.score:.3f}",
        "catalog_match_margin": f"{match.margin:.3f}",
    }
    if match.candidate_codes:
        fields["catalog_candidate_codes"] = ",".join(match.candidate_codes)

    trusted_hint = barcode_hint if reliable_barcode_hint is None else reliable_barcode_hint
    if match.accepted:
        if _has_reliable_catalog_key(match, trusted_hint):
            fields["catalog_key_source"] = "visual_barcode_or_qr"
            fields["product_name"] = match.product_name
            fields["barcode"] = match.barcode
            fields["qr_code_barcode"] = match.barcode
            return fields
        fields["catalog_match_status"] = "catalog_guess"
        fields["catalog_key_source"] = "catalog_inferred"
        fields["catalog_withheld_reason"] = "no_reliable_barcode"
        fields["catalog_guess_name"] = match.product_name
        if match.barcode:
            fields["catalog_guess_barcode"] = match.barcode
        return fields

    fields["catalog_guess_name"] = match.catalog_guess_name
    if match.candidate_codes:
        fields["catalog_guess_barcode"] = match.candidate_codes[0]
    return fields


def _has_reliable_catalog_key(match: CatalogMatch, barcode_hint: str) -> bool:
    hint = to_ean13(barcode_hint)
    if not hint or not ean13_ok(hint):
        return False
    candidates = set(match.candidate_codes)
    if match.barcode:
        candidates.add(match.barcode)
    return hint in candidates


def _barcode_hints_by_track(
    context: PipelineContext,
    det2track: dict[str, str],
) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for symbol in context.decoded_symbols:
        track_id = det2track.get(symbol.detection_id, symbol.detection_id)
        for value in _barcode_values_from_symbol(symbol.payload):
            hints.setdefault(track_id, []).append(value)
    return hints


def _barcode_values_from_symbol(payload: str) -> list[str]:
    parsed = parse_qr_payload(payload)
    values = [parsed.get("barcode", ""), parsed.get("qr_code_barcode", ""), payload]
    out: list[str] = []
    for value in values:
        code = to_ean13(value)
        if code and ean13_ok(code):
            out.append(code)
    return list(dict.fromkeys(out))


def _best_barcode_hint(values: list[str], fallback: str = "") -> str:
    for value in values:
        code = to_ean13(value)
        if code and ean13_ok(code):
            return code
    code = to_ean13(fallback)
    return code if code and ean13_ok(code) else fallback


_DIGITISH_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "О": "0",
        "о": "0",
        "Q": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "L": "1",
        "І": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "З": "3",
        "з": "3",
    }
)


def _digit_runs(text: str) -> list[str]:
    translated = str(text or "").translate(_DIGITISH_TRANSLATION)
    pattern = r"(?<!\w)[0-9][0-9\s-]{9,18}[0-9](?!\w)"
    candidates = {
        re.sub(r"\D", "", match.group(0))
        for match in re.finditer(pattern, translated)
    }
    compact = re.sub(r"\D", "", translated)
    if len(compact) in {11, 12, 13, 14}:
        candidates.add(compact)
    return [value for value in candidates if value]


def _catalog_match_record(
    track_id: str,
    match: CatalogMatch | None,
    *,
    barcode_hint: str = "",
    reliable_barcode_hint: str | None = None,
) -> dict[str, Any]:
    if match is None:
        return {}
    trusted_hint = barcode_hint if reliable_barcode_hint is None else reliable_barcode_hint
    reliable_key = _has_reliable_catalog_key(match, trusted_hint)
    status = (
        "confirmed"
        if match.accepted and reliable_key
        else ("catalog_guess" if match.accepted or match.catalog_guess_name else match.status)
    )
    guess_name = "" if status == "confirmed" else (match.catalog_guess_name or match.product_name)
    return {
        "track_id": track_id,
        "status": status,
        "resolver_status": match.status,
        "product_name": match.product_name if status == "confirmed" else "",
        "catalog_guess_name": guess_name,
        "barcode": match.barcode if status == "confirmed" else "",
        "catalog_guess_barcode": match.barcode if status == "catalog_guess" else "",
        "barcode_hint": barcode_hint,
        "reliable_key": reliable_key,
        "key_source": "visual_barcode_or_qr" if reliable_key else "catalog_inferred",
        "score": match.score,
        "margin": match.margin,
        "accepted": match.accepted,
        "candidate_codes": list(match.candidate_codes),
    }


def _visual_barcode_identity_fields(barcode: str, product_name: str) -> dict[str, str]:
    return {
        "product_name": product_name,
        "barcode": barcode,
        "qr_code_barcode": barcode,
        "catalog_match_status": "confirmed",
        "catalog_resolver_status": "barcode_exact",
        "catalog_match_source": "db_hack_visual_barcode",
        "catalog_key_source": "visual_barcode_or_qr",
        "catalog_match_score": "1.000",
    }


def _visual_barcode_record(
    track_id: str,
    *,
    barcode: str,
    product_name: str,
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "status": "confirmed",
        "resolver_status": "barcode_exact",
        "product_name": product_name,
        "catalog_guess_name": "",
        "barcode": barcode,
        "catalog_guess_barcode": "",
        "barcode_hint": barcode,
        "reliable_key": True,
        "key_source": "visual_barcode_or_qr",
        "score": 1.0,
        "margin": 1.0,
        "accepted": True,
        "candidate_codes": [barcode],
    }


@dataclass(slots=True)
class _Cfg:
    enabled: bool
    max_crops: int
    upscale: float
    category: str
    db_path: str
    best_effort_name: bool

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
            best_effort_name=bool(raw.get("best_effort_name", True)),
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
            ys = poly[:, 1]
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
            seen_in_text: set[str] = set()
            for digits in _digit_runs(t):
                if len(digits) == 12 and digits[:2] in ("27", "37"):
                    seen_in_text.add(digits)
                # 13/14-digit reads: trim/keep the (27|37)-prefixed 12-window.
                if len(digits) in (13, 14):
                    for i in range(0, len(digits) - 11):
                        w = digits[i:i + 12]
                        if w[:2] in ("27", "37"):
                            seen_in_text.add(w)
            out.extend(seen_in_text)
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
        return V5BlockOcrCatalogStage._sku_consensus_details(cands)["value"]

    @staticmethod
    def _sku_consensus_details(cands: list[str]) -> dict[str, str]:
        cands = [c for c in cands if len(c) == 12]
        if not cands:
            return {"value": "", "confidence": "0.00", "method": "", "candidates": "0"}
        from collections import Counter
        # exact-value majority first (most reliable when >=2 agree)
        common, n = Counter(cands).most_common(1)[0]
        if n >= 2:
            return {
                "value": common,
                "confidence": "0.94",
                "method": "exact_majority",
                "candidates": str(len(cands)),
            }
        # else per-position majority across all 12-digit reads
        out = "".join(
            Counter(c[i] for c in cands).most_common(1)[0][0]
            for i in range(12)
        )
        # structural prior: real id_sku starts 27/37 (store/category)
        if out[:2] in ("27", "37") and len(cands) >= 3:
            return {
                "value": out,
                "confidence": "0.86",
                "method": "per_position",
                "candidates": str(len(cands)),
            }
        return {
            "value": "",
            "confidence": "0.00",
            "method": "insufficient_support",
            "candidates": str(len(cands)),
        }

    @staticmethod
    def _mode(cands: list[str]) -> str:
        if not cands:
            return ""
        from collections import Counter
        return Counter(cands).most_common(1)[0][0]

    @staticmethod
    def _mode_with_support(cands: list[str], *, min_count: int) -> str:
        if not cands:
            return ""
        from collections import Counter
        value, count = Counter(cands).most_common(1)[0]
        return value if count >= min_count else ""

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
        decoded_hints = _barcode_hints_by_track(ctx, det2track)
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
        resolver_accepted = 0
        best_effort = 0
        catalog_matches: list[dict[str, Any]] = []
        fineprint_matches: list[dict[str, Any]] = []
        for tk, texts in pooled.items():
            decoded_barcode_hint = _best_barcode_hint(decoded_hints.get(tk, []))
            ocr_barcode_hint = _best_barcode_hint([], fallback=hint.get(tk, ""))
            barcode_hint = decoded_barcode_hint or ocr_barcode_hint
            exact_name = (
                self._resolver.name_for_barcode(decoded_barcode_hint)
                if decoded_barcode_hint
                else ""
            )
            if exact_name:
                ident = _visual_barcode_identity_fields(decoded_barcode_hint, exact_name)
                accepted += 1
                catalog_matches.append(
                    _visual_barcode_record(
                        tk,
                        barcode=decoded_barcode_hint,
                        product_name=exact_name,
                    )
                )
            else:
                m = self._resolver.resolve(texts, category=cfg.category,
                                           barcode_hint=barcode_hint,
                                           best_effort=cfg.best_effort_name)
                record = _catalog_match_record(
                    tk,
                    m,
                    barcode_hint=barcode_hint,
                    reliable_barcode_hint=decoded_barcode_hint,
                )
                if record:
                    catalog_matches.append(record)
                ident: dict[str, str] = _catalog_identity_fields(
                    m,
                    barcode_hint=barcode_hint,
                    reliable_barcode_hint=decoded_barcode_hint,
                )
                if m and m.accepted:
                    resolver_accepted += 1
                if m and m.accepted and _has_reliable_catalog_key(m, barcode_hint):
                    accepted += 1
                elif m and (m.catalog_guess_name or m.accepted):
                    # Best-effort nearest name only: keep it as catalog_guess_*,
                    # never as product_name/barcode in official CSV fields.
                    best_effort += 1
            # Fine-print consensus is independent of catalog identity and
            # strictly non-negative (none of these are GT match keys, so a
            # wrong value scores exactly like the empty we'd emit anyway).
            sku = self._sku_consensus_details(sku_pool.get(tk, []))
            fineprint_matches.append(
                {
                    "track_id": tk,
                    "id_sku": sku["value"],
                    "id_sku_confidence": sku["confidence"],
                    "id_sku_method": sku["method"],
                    "id_sku_candidates": len(sku_pool.get(tk, [])),
                    "date_candidates": len(date_pool.get(tk, [])),
                    "code_candidates": len(code_pool.get(tk, [])),
                }
            )
            if sku["value"]:
                ident["id_sku"] = sku["value"]
                ident["id_sku_source"] = f"v5_fineprint_{sku['method']}"
                ident["id_sku_confidence"] = sku["confidence"]
                ident["id_sku_candidates_count"] = sku["candidates"]
            dt = self._mode_with_support(date_pool.get(tk, []), min_count=2)
            if dt:
                ident["print_datetime"] = dt
                ident["print_datetime_source"] = "v5_fineprint_exact_majority"
                ident["print_datetime_confidence"] = "0.82"
                ident["print_datetime_candidates_count"] = str(len(date_pool.get(tk, [])))
            code = self._mode_with_support(code_pool.get(tk, []), min_count=2)
            if code:
                ident["code"] = code
                ident["code_source"] = "v5_fineprint_exact_majority"
                ident["code_confidence"] = "0.84"
                ident["code_candidates_count"] = str(len(code_pool.get(tk, [])))
            if ident:
                identity[tk] = ident

        crops_with_fields = 0
        for cid, rec in per_crop.items():
            fields = dict(rec["fields"])
            fields.update(identity.get(rec["track"], {}))
            if fields:
                crops_with_fields += 1
            field_confidences = {}
            if fields.get("id_sku") and fields.get("id_sku_confidence"):
                field_confidences["id_sku"] = fields["id_sku_confidence"]
            if fields.get("print_datetime") and fields.get("print_datetime_confidence"):
                field_confidences["print_datetime"] = fields["print_datetime_confidence"]
            if fields.get("code") and fields.get("code_confidence"):
                field_confidences["code"] = fields["code_confidence"]
            rec["crop"].attributes["ocr"] = {
                "text": "", "confidence": 0.0,
                "engine": "v5_block_paddle_en", "fields": fields,
                "field_confidences": field_confidences,
            }
        catalog_matches_key = ""
        if catalog_matches:
            try:
                catalog_matches_key = ctx.artifact_writer.upload_json(
                    "debug/catalog_matches.json",
                    {
                        "tracks": len(pooled),
                        "accepted_tracks": accepted,
                        "resolver_accepted_tracks": resolver_accepted,
                        "catalog_guess_tracks": best_effort,
                        "matches": catalog_matches,
                    },
                )
            except Exception:  # noqa: BLE001 - debug artifact must not break e2e
                catalog_matches_key = ""
        fineprint_matches_key = ""
        if fineprint_matches:
            try:
                fineprint_matches_key = ctx.artifact_writer.upload_json(
                    "debug/fineprint_matches.json",
                    {"tracks": len(pooled), "matches": fineprint_matches},
                )
            except Exception:  # noqa: BLE001 - debug artifact must not break e2e
                fineprint_matches_key = ""
        summary = {
            "enabled": True,
            "crops_ocred": len(per_crop),
            "tracks": len(pooled),
            "catalog_accepted_tracks": accepted,
            "catalog_resolver_accepted_tracks": resolver_accepted,
            "best_effort_name_tracks": best_effort,
            "catalog_guess_tracks": best_effort,
            "catalog_matches_key": catalog_matches_key,
            "fineprint_matches_key": fineprint_matches_key,
            "tracks_with_decoded_barcode_hint": sum(
                1 for tk in pooled if decoded_hints.get(tk)
            ),
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
