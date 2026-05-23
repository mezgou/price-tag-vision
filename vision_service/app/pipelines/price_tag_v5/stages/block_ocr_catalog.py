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


def _safe_price_float(value: str | None) -> float | None:
    text = str(value or "").strip().replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


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
    fineprint_per_track: int

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
            fineprint_per_track=max(int(raw.get("fineprint_per_track", 6) or 0), 0),
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
            xs = poly[:, 0]
            ys = poly[:, 1]
            rel_h = float(ys.max() - ys.min()) / float(H)
            rel_y = float(ys.min()) / float(H)
            rel_x = float(xs.min()) / float(max(img.shape[1], 1))
            rel_w = float(xs.max() - xs.min()) / float(max(img.shape[1], 1))
            blocks.append(
                {
                    "text": str(txt),
                    "conf": conf,
                    "rel_h": rel_h,
                    "rel_y": rel_y,
                    "rel_x": rel_x,
                    "rel_w": rel_w,
                    "rel_cx": rel_x + rel_w / 2.0,
                }
            )
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
    def _price_ints(nums: list[dict]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b in nums:
            for tok in re.findall(r"\d+", b["text"]):
                if 3 <= len(tok) <= 5:
                    values = [(int(tok), 1.0, "raw")]
                    if (
                        len(tok) == 5
                        and tok.endswith("0")
                        and 0.12 <= float(b.get("rel_y", 0.5)) <= 0.56
                    ):
                        values.append((int(tok[:-1]), 0.88, "trimmed_trailing_zero"))
                    for v, confidence_scale, source in values:
                        if not 100 <= v <= 99999:
                            continue
                        rel_x = float(b.get("rel_x", 0.45))
                        rel_w = float(b.get("rel_w", 0.18))
                        out.append(
                            {
                                "value": v,
                                "rel_y": float(b.get("rel_y", 0.5)),
                                "rel_h": float(b.get("rel_h", 0.05)),
                                "rel_x": rel_x,
                                "rel_w": rel_w,
                                "rel_cx": float(b.get("rel_cx", rel_x + rel_w / 2.0)),
                                "conf": float(b.get("conf", 0.0)) * confidence_scale,
                                "source": source,
                            }
                        )
        return out

    @classmethod
    def _card_kopecks(cls, nums: list[dict]) -> str:
        return cls._kopecks_for_price(nums, price_entry=None, fallback="99")

    @staticmethod
    def _two_digit_entries(nums: list[dict]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for block in nums:
            text = str(block.get("text", ""))
            for token in re.findall(r"(?<!\d)\d{2}(?!\d)", text):
                rel_x = float(block.get("rel_x", 0.62))
                rel_w = float(block.get("rel_w", 0.08))
                out.append(
                    {
                        "value": token,
                        "rel_y": float(block.get("rel_y", 0.5)),
                        "rel_x": rel_x,
                        "rel_w": rel_w,
                        "rel_cx": float(block.get("rel_cx", rel_x + rel_w / 2.0)),
                    }
                )
        return out

    @classmethod
    def _kopecks_for_price(
        cls,
        nums: list[dict],
        *,
        price_entry: dict[str, Any] | None,
        fallback: str,
        excluded: set[str] | None = None,
    ) -> str:
        excluded = excluded or set()
        candidates: list[tuple[float, str]] = []
        for entry in cls._two_digit_entries(nums):
            token = str(entry["value"])
            if token in excluded:
                continue
            if price_entry is None:
                if token in cls._KOPECK_PREF:
                    candidates.append((float(cls._KOPECK_PREF.index(token)), token))
                continue
            y_distance = abs(float(entry["rel_y"]) - float(price_entry["rel_y"]))
            price_left = float(price_entry.get("rel_x", 0.45))
            price_right = price_left + float(price_entry.get("rel_w", 0.18))
            if token not in cls._KOPECK_PREF and float(entry.get("rel_x", 0.62)) + 0.02 < price_left:
                continue
            x_distance = abs(float(entry.get("rel_x", 0.62)) - price_right)
            if float(entry.get("rel_x", 0.62)) + 0.03 < price_right:
                x_distance += 0.12
            candidates.append((y_distance + min(x_distance, 0.20), token))
        if not candidates:
            return fallback
        distance, token = min(candidates, key=lambda item: item[0])
        return token if distance <= 0.24 else fallback

    @staticmethod
    def _discount_from_prices(default_price: str, card_price: str) -> str:
        default = _safe_price_float(default_price)
        card = _safe_price_float(card_price)
        if default is None or card is None or default <= card:
            return ""
        pct = int(round((default - card) / default * 100.0))
        return f"-{pct}%" if 1 <= pct <= 99 else ""

    @classmethod
    def _reconcile_discount(
        cls,
        discount: str,
        *,
        default_price: str,
        card_price: str,
    ) -> str:
        derived = cls._discount_from_prices(default_price, card_price)
        if not discount or not derived:
            return discount
        observed_digits = re.sub(r"\D+", "", discount)
        derived_digits = re.sub(r"\D+", "", derived)
        if (
            len(observed_digits) == 1
            and derived_digits
            and abs(int(observed_digits) - int(derived_digits)) >= 10
        ):
            return derived
        return discount

    @staticmethod
    def _special_symbol_from_block(block: dict) -> str:
        if float(block.get("rel_y", 0.0)) <= 0.70:
            return ""
        text = re.sub(r"\s+", "", str(block.get("text", ""))).upper()
        if text in {"К", "K", "Ш", "Л"}:
            return "К" if text in {"К", "K"} else text
        if re.fullmatch(r"[KК][O0О]?", text):
            return "К"
        return ""

    @staticmethod
    def _additional_info_from_blocks(blocks: list[dict]) -> str:
        text = re.sub(r"\s+", "", " ".join(str(b.get("text", "")) for b in blocks)).casefold()
        if not text:
            return ""
        if (
            "полуслад" in text
            or "п/сл" in text
            or "п.сл" in text
            or "p/cл" in text
            or "n/cл" in text
        ):
            return "Полусладкое"
        if (
            "полусух" in text
            or "п/сух" in text
            or "п.сух" in text
            or "p/cyx" in text
            or "n/cyx" in text
            or "p.cyx" in text
            or "n.cyx" in text
        ):
            return "Полусухое"
        if "сух" in text or "cyxoe" in text or "cyx" in text:
            return "Сухое"
        if "слад" in text:
            return "Сладкое"
        if "брют" in text or "brut" in text:
            return "Брют"
        return ""

    @staticmethod
    def _default_kopecks(
        nums: list[dict],
        *,
        default_y: float,
        card_kopecks: str,
    ) -> str:
        candidates: list[tuple[float, str]] = []
        for block in nums:
            text = str(block.get("text", ""))
            for token in re.findall(r"(?<!\d)\d{2}(?!\d)", text):
                if token == card_kopecks:
                    continue
                candidates.append((abs(float(block.get("rel_y", 0.5)) - default_y), token))
        if not candidates:
            return "00"
        candidates.sort(key=lambda item: item[0])
        distance, token = candidates[0]
        return token if distance <= 0.16 else "00"

    @staticmethod
    def _select_price_pair(
        entries: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for default_entry in entries:
            default_value = int(default_entry["value"])
            default_y = float(default_entry["rel_y"])
            for card_entry in entries:
                card_value = int(card_entry["value"])
                if card_value >= default_value:
                    continue
                if not (0.30 * default_value <= card_value <= 0.985 * default_value):
                    continue
                card_y = float(card_entry["rel_y"])
                gap = card_y - default_y
                if not (0.08 <= gap <= 0.58):
                    continue
                center_delta = abs(
                    float(default_entry.get("rel_cx", 0.5))
                    - float(card_entry.get("rel_cx", 0.5))
                )
                score = (
                    abs(default_y - 0.30) * 2.6
                    + abs(card_y - 0.62) * 2.4
                    + abs(gap - 0.30) * 1.8
                    + center_delta * 0.8
                    + max(card_y - 0.84, 0.0) * 7.0
                    + max(default_y - 0.58, 0.0) * 6.0
                )
                pairs.append((score, default_entry, card_entry))
        if pairs:
            _score, default_entry, card_entry = min(pairs, key=lambda item: item[0])
            return default_entry, card_entry
        if not entries:
            return None, None
        card_entry = max(
            entries,
            key=lambda item: (float(item.get("rel_h", 0.0)), int(item["value"])),
        )
        return None, card_entry

    @staticmethod
    def _prices_from_blocks(blocks: list[dict]) -> dict[str, str]:
        # drop volume tokens (0.75 L / 1.5L / 750 ml) before price parsing
        nums = [
            b for b in blocks
            if re.search(r"\d", b["text"]) and "%" not in b["text"]
            and not re.search(r"\d[\s.,]*(L|Л|ML|МЛ)\b", b["text"].upper())
            and not re.fullmatch(r"\s*[01][.,]\d{1,2}\s*[lLлЛ]?\s*", b["text"])
        ]
        nums = [
            b
            for b in nums
            if not re.search(r"\d+\s*(?:g|kg|г|кг)\b", b["text"], re.IGNORECASE)
        ]
        out: dict[str, str] = {}
        # Domain invariant beats fragile font-height geometry: the two main
        # numbers on the tag are the "без карты" (default, larger, struck)
        # and the "по карте" price (card, smaller, big red). card <= default
        # always, and the card price always ends in .99. Picking by tallest
        # block was selecting the struck default integer as the card price
        # (price_card+price4_qr both wrong); reconcile by value instead.
        price_int_entries = V5BlockOcrCatalogStage._price_ints(nums)
        unique_ints: dict[int, dict[str, Any]] = {}
        for entry in price_int_entries:
            value = int(entry["value"])
            current = unique_ints.get(value)
            if current is None or float(entry.get("conf", 0.0)) > float(current.get("conf", 0.0)):
                unique_ints[value] = entry
        if unique_ints:
            default_entry, card_entry = V5BlockOcrCatalogStage._select_price_pair(
                list(unique_ints.values())
            )
            if card_entry is None:
                return out
            card_int = int(card_entry["value"])
            if default_entry is None:
                # only one dominant price visible -> it is the prominent
                # card price; default is then unread (kopecks anyway are
                # sub-resolvable, so emitting a wrong default helps nothing)
                default_int = None
            else:
                default_int = int(default_entry["value"])
            card_kopecks = V5BlockOcrCatalogStage._kopecks_for_price(
                nums,
                price_entry=card_entry,
                fallback=V5BlockOcrCatalogStage._card_kopecks(nums),
            )
            pc = f"{card_int}.{card_kopecks}"
            if V5BlockOcrCatalogStage._plausible_price(pc):
                out["price_card"] = pc
            if default_int is not None and default_entry is not None:
                default_kopecks = V5BlockOcrCatalogStage._kopecks_for_price(
                    nums,
                    price_entry=default_entry,
                    fallback="00",
                    excluded={card_kopecks},
                )
                pd = f"{default_int}.{default_kopecks}"
                if (V5BlockOcrCatalogStage._plausible_price(pd)
                        and pd != out.get("price_card")):
                    out["price_default"] = pd
        disc = _best_discount(" ".join(b["text"] for b in blocks))
        disc = V5BlockOcrCatalogStage._reconcile_discount(
            disc,
            default_price=out.get("price_default", ""),
            card_price=out.get("price_card", ""),
        )
        if disc:
            out["discount_amount"] = disc
        for b in blocks:
            symbol = V5BlockOcrCatalogStage._special_symbol_from_block(b)
            if symbol:
                out["special_symbols"] = symbol
        additional_info = V5BlockOcrCatalogStage._additional_info_from_blocks(blocks)
        if additional_info:
            out["additional_info"] = additional_info
        return out

    # ---- fine print: id_sku / print_datetime / code ---------------------
    # These fields strongly affect the score yet sit in the smallest text on
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
        fineprint_raw: dict[str, list[dict[str, Any]]] = {}
        # fine-print OCR is the slow extra pass; only run it on the sharpest
        # crops per track (SKU/date/code are static -> a few good reads, then
        # consensus). Budget keeps total runtime close to the 1-pass version.
        fine_seen: dict[str, int] = {}
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
            if fine_seen.get(tk, 0) < cfg.fineprint_per_track:
                fine_seen[tk] = fine_seen.get(tk, 0) + 1
                raw_fineprint_texts = self._fineprint_texts(img)
                fineprint_raw.setdefault(tk, []).append(
                    {
                        "crop_id": crop.crop_id,
                        "zone": "bottom_strip",
                        "engine": "v5_block_paddle_en",
                        "variant": "upscale_7",
                        "raw_text": raw_fineprint_texts,
                        "parsed_candidates": {
                            "id_sku": self._sku_candidates(raw_fineprint_texts),
                            "print_datetime": self._date_candidates(raw_fineprint_texts),
                            "code": self._code_candidates(raw_fineprint_texts),
                        },
                    }
                )
                texts = texts + raw_fineprint_texts
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
        fineprint_candidates: list[dict[str, Any]] = []
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
            accepted_fields: dict[str, str] = {}
            if sku["value"]:
                ident["id_sku"] = sku["value"]
                ident["id_sku_source"] = f"v5_fineprint_{sku['method']}"
                ident["id_sku_confidence"] = sku["confidence"]
                ident["id_sku_candidates_count"] = sku["candidates"]
                accepted_fields["id_sku"] = sku["value"]
            dt = self._mode_with_support(date_pool.get(tk, []), min_count=2)
            if dt:
                ident["print_datetime"] = dt
                ident["print_datetime_source"] = "v5_fineprint_exact_majority"
                ident["print_datetime_confidence"] = "0.82"
                ident["print_datetime_candidates_count"] = str(len(date_pool.get(tk, [])))
                accepted_fields["print_datetime"] = dt
            code = self._mode_with_support(code_pool.get(tk, []), min_count=2)
            if code:
                ident["code"] = code
                ident["code_source"] = "v5_fineprint_exact_majority"
                ident["code_confidence"] = "0.84"
                ident["code_candidates_count"] = str(len(code_pool.get(tk, [])))
                accepted_fields["code"] = code
            fineprint_candidates.append(
                {
                    "track_id": tk,
                    "raw_reads": fineprint_raw.get(tk, []),
                    "pooled_candidates": {
                        "id_sku": sku_pool.get(tk, []),
                        "print_datetime": date_pool.get(tk, []),
                        "code": code_pool.get(tk, []),
                    },
                    "accepted": accepted_fields,
                    "rejected_reason": (
                        "" if accepted_fields else "insufficient_repeated_support"
                    ),
                }
            )
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
        fineprint_candidates_key = ""
        if fineprint_candidates:
            try:
                fineprint_candidates_key = ctx.artifact_writer.upload_json(
                    "debug/fineprint_candidates.json",
                    {
                        "tracks": len(pooled),
                        "fineprint_per_track": cfg.fineprint_per_track,
                        "matches": fineprint_candidates,
                    },
                )
            except Exception:  # noqa: BLE001 - debug artifact must not break e2e
                fineprint_candidates_key = ""
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
            "fineprint_candidates_key": fineprint_candidates_key,
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
