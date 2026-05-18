from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.pipelines.base import BaseStage, PipelineContext, SampledFrameMetadata, StageOutcome
from app.schemas.detections import CropCandidate
from app.utils.image_processing import clip_bbox_to_frame

NO_VALUE = "нет"
MOJIBAKE_NO_VALUE = "РЅРµС‚"
EAN13_RE = re.compile(r"\b\d(?:[\s-]?\d){12}\b")
SKU_RE = re.compile(r"\b\d(?:[\s-]?\d){5,11}\b")
PRICE_RE = re.compile(r"(?<!\d)(\d{1,5})\s*[,.\s]\s*(\d{2})(?!\d)")
DISCOUNT_RE = re.compile(r"[-−]\s*\d{1,2}\s*%")
DATE_RE = re.compile(
    r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}(?:\s+\d{1,2}:\d{2})?\b"
)
PRODUCT_STOPWORDS = {
    "руб",
    "карта",
    "картой",
    "цена",
    "скидка",
    "сухое",
    "полусухое",
    "полусладкое",
    "сладкое",
}


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError


@dataclass(slots=True)
class ZonalOcrConfig:
    enabled: bool
    max_crops: int
    min_crop_quality_score: float
    engine_order: list[str]
    tesseract_lang: str
    warnings: list[str]

    @classmethod
    def from_context(cls, context: PipelineContext) -> "ZonalOcrConfig":
        raw_config = context.config.get("zonal_ocr", {})
        warnings: list[str] = []
        if not isinstance(raw_config, dict):
            warnings.append("zonal_ocr config must be a mapping. Falling back to defaults.")
            raw_config = {}
        engine_order = raw_config.get("engine_order", ["paddle", "rapidocr", "tesseract"])
        if not isinstance(engine_order, list):
            warnings.append("zonal_ocr.engine_order must be a list. Falling back to defaults.")
            engine_order = ["paddle", "rapidocr", "tesseract"]
        return cls(
            enabled=_coerce_bool(raw_config.get("enabled"), default=True),
            max_crops=_coerce_positive_int(
                raw_config.get("max_crops"),
                default=120,
                field_name="zonal_ocr.max_crops",
                warnings=warnings,
            ),
            min_crop_quality_score=_coerce_ratio_inclusive_zero(
                raw_config.get("min_crop_quality_score"),
                default=0.08,
                field_name="zonal_ocr.min_crop_quality_score",
                warnings=warnings,
            ),
            engine_order=[str(item).strip().lower() for item in engine_order if str(item).strip()],
            tesseract_lang=str(raw_config.get("tesseract_lang") or "rus+eng"),
            warnings=warnings,
        )


class ZonalOcrStage(BaseStage):
    name = "ZonalOcrStage"

    def __init__(self) -> None:
        self._engine_cache: dict[str, OcrEngine | None] = {}

    def describe_input(self, context: PipelineContext) -> dict[str, Any]:
        config = ZonalOcrConfig.from_context(context)
        return {
            "enabled": config.enabled,
            "crops_count": len(context.crop_candidates),
            "max_crops": config.max_crops,
            "min_crop_quality_score": config.min_crop_quality_score,
            "engine_order": config.engine_order,
        }

    def run(self, context: PipelineContext) -> StageOutcome:
        config = ZonalOcrConfig.from_context(context)
        warnings = list(config.warnings)
        if not config.enabled or not context.crop_candidates:
            context.artifacts["ocr_summary"] = {
                "enabled": config.enabled,
                "crops_processed": 0,
                "crops_with_fields": 0,
            }
            return StageOutcome(output_summary=context.artifacts["ocr_summary"], warnings=warnings)

        frame_lookup = _build_frame_lookup(context.sampled_frames)
        eligible_crops = [
            crop
            for crop in sorted(
                context.crop_candidates,
                key=lambda item: item.quality.score,
                reverse=True,
            )
            if crop.quality.score >= config.min_crop_quality_score
        ][: config.max_crops]

        engines = self._resolve_engines(config=config, warnings=warnings)
        crops_processed = 0
        crops_with_text = 0
        crops_with_fields = 0
        engine_usage: Counter[str] = Counter()

        for crop in eligible_crops:
            crop_image = _load_crop_image(crop=crop, frame_lookup=frame_lookup)
            if crop_image is None:
                warnings.append(f"Crop {crop.crop_id} could not be reconstructed for OCR.")
                continue

            crops_processed += 1
            color = classify_tag_color(crop_image)
            zonal = _recognize_zonal(crop_image, engines=engines)
            fields = _fields_from_zones(zonal.zone_texts)
            if not fields.get("product_name") or not fields.get("barcode"):
                # belt-and-suspenders whole-crop pass for the hard fields
                blob = parse_ocr_fields(zonal.combined_text)
                for k, v in blob.items():
                    fields.setdefault(k, v)
            if color:
                fields["color"] = color
            if zonal.combined_text:
                crops_with_text += 1
                engine_usage[zonal.engine_name] += 1
            if fields:
                crops_with_fields += 1

            crop.attributes["ocr"] = {
                "text": zonal.combined_text,
                "confidence": round(zonal.confidence, 6),
                "engine": zonal.engine_name,
                "orientation": "zonal",
                "fields": fields,
            }

        summary = {
            "enabled": True,
            "eligible_crops_count": len(eligible_crops),
            "crops_processed": crops_processed,
            "crops_with_text": crops_with_text,
            "crops_with_fields": crops_with_fields,
            "engine_usage": dict(engine_usage),
        }
        context.artifacts["ocr_summary"] = summary
        return StageOutcome(output_summary=summary, warnings=warnings)

    def _resolve_engines(
        self,
        *,
        config: ZonalOcrConfig,
        warnings: list[str],
    ) -> list[OcrEngine]:
        engines: list[OcrEngine] = []
        for engine_name in config.engine_order:
            if engine_name not in self._engine_cache:
                self._engine_cache[engine_name] = _create_engine(
                    engine_name,
                    config=config,
                    warnings=warnings,
                )
            engine = self._engine_cache[engine_name]
            if engine is not None:
                engines.append(engine)
        return engines


@dataclass(frozen=True, slots=True)
class _RecognitionResult:
    text: str
    confidence: float
    engine_name: str
    orientation: str


class _PaddleOcrEngine:
    name = "paddle"

    @cached_property
    def engine(self) -> Any:
        # torch MUST be imported before paddleocr: paddleocr -> ppocr ->
        # albumentations -> albumentations.pytorch -> torch; importing torch
        # first makes the cached module satisfy the late import (Windows
        # shm.dll WinError 127 otherwise).
        import torch  # noqa: F401
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        # The Russian PP-OCR model is broken in this paddleocr 2.10 / paddle
        # 3.0.0 stack (returns confident garbage even on flawless digits).
        # The English model works perfectly and covers every numeric field
        # (prices, discount, barcode, SKU, date) which carry the metric.
        for kwargs in (
            {"lang": "en", "use_textline_orientation": True},
            {"lang": "en", "use_angle_cls": True},
            {"lang": "en"},
        ):
            try:
                return PaddleOCR(**kwargs)
            except Exception:  # noqa: BLE001
                continue
        return PaddleOCR(lang="en")

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        engine = self.engine
        result: Any
        if hasattr(engine, "predict"):
            result = engine.predict(image)
        else:
            result = engine.ocr(image)
        texts: list[str] = []
        confidences: list[float] = []
        for page in result or []:
            # paddleocr 3.x: page is a dict with rec_texts / rec_scores
            if isinstance(page, dict):
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                for idx, text in enumerate(rec_texts):
                    normalized = str(text).strip()
                    if not normalized:
                        continue
                    texts.append(normalized)
                    if idx < len(rec_scores):
                        confidences.append(_safe_float(rec_scores[idx], default=0.0))
                continue
            # legacy 2.x: page is a list of [box, (text, score)]
            for item in page or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                payload = item[1]
                if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                    continue
                text = str(payload[0]).strip()
                if not text:
                    continue
                texts.append(text)
                confidences.append(_safe_float(payload[1], default=0.0))
        return "\n".join(texts), _mean(confidences)


class _RapidOcrEngine:
    name = "rapidocr"

    @cached_property
    def engine(self) -> Any:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        return RapidOCR()

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        result, _elapsed = self.engine(image)
        texts: list[str] = []
        confidences: list[float] = []
        for item in result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            texts.append(text)
            confidences.append(_safe_float(item[2], default=0.0))
        return "\n".join(texts), _mean(confidences)


class _TesseractOcrEngine:
    name = "tesseract"

    def __init__(self, *, lang: str) -> None:
        self.lang = lang

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config="--psm 6",
            output_type=Output.DICT,
        )
        texts: list[str] = []
        confidences: list[float] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", [])):
            normalized = str(text).strip()
            if not normalized:
                continue
            texts.append(normalized)
            conf = _safe_float(confidence, default=-1.0)
            if conf >= 0:
                confidences.append(conf / 100.0)
        return " ".join(texts), _mean(confidences)


def _create_engine(
    engine_name: str,
    *,
    config: ZonalOcrConfig,
    warnings: list[str],
) -> OcrEngine | None:
    try:
        if engine_name == "paddle":
            return _PaddleOcrEngine()
        if engine_name == "rapidocr":
            return _RapidOcrEngine()
        if engine_name == "tesseract":
            return _TesseractOcrEngine(lang=config.tesseract_lang)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"OCR engine '{engine_name}' is unavailable: {exc}")
        return None
    warnings.append(f"Unknown OCR engine '{engine_name}' was skipped.")
    return None


def _recognize_best_orientation(
    crop: np.ndarray,
    *,
    engines: list[OcrEngine],
) -> _RecognitionResult:
    best = _RecognitionResult(text="", confidence=0.0, engine_name="", orientation="none")
    if not engines:
        return best

    for orientation, oriented in _orientation_variants(crop).items():
        prepared = _prepare_for_ocr(oriented)
        for engine in engines:
            try:
                text, confidence = engine.recognize(prepared)
            except Exception:
                continue
            score = _text_score(text=text, confidence=confidence)
            if score > _text_score(text=best.text, confidence=best.confidence):
                best = _RecognitionResult(
                    text=_normalize_ocr_text(text),
                    confidence=confidence,
                    engine_name=engine.name,
                    orientation=orientation,
                )
            if best.confidence >= 0.82 and len(best.text) >= 20:
                return best
    return best


# Tag-relative zones from executive_summary 7.2 (x1,y1,x2,y2 fractions OF THE
# TAG). The crop is the detector bbox padded by ~12% per side, so the tag
# occupies ~[0.097, 0.903] of the crop; _zone_box() maps tag->crop fractions.
# Derived from the real red alcohol-tag layout (verified on close crops):
# white upper third = name (+ QR right, small no-card price right); red lower
# two-thirds = big card price (center-right), discount circle (left), barcode
# (bottom). The doc 7.2 fractions were too tight/high for these tags.
_TAG_ZONES: dict[str, tuple[float, float, float, float]] = {
    "product_name": (0.00, 0.00, 0.66, 0.34),
    "price_default": (0.60, 0.24, 1.00, 0.48),
    "price_card": (0.22, 0.44, 1.00, 0.92),
    "discount": (0.00, 0.48, 0.34, 0.92),
    "additional_info": (0.14, 0.28, 0.62, 0.50),
    "sku_date": (0.00, 0.78, 0.58, 1.00),
    "barcode": (0.34, 0.82, 1.00, 1.00),
    "special": (0.26, 0.83, 0.48, 0.99),
}
_PAD_LO = 0.097
_PAD_SPAN = 0.806


@dataclass(frozen=True, slots=True)
class _ZonalResult:
    zone_texts: dict[str, str]
    combined_text: str
    confidence: float
    engine_name: str


def _zone_box(
    frac: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = frac
    # tag-fraction -> padded-crop-fraction, with a little outward slack
    def m(v: float) -> float:
        return _PAD_LO + _PAD_SPAN * v

    slack = 0.025
    cx1 = max(0.0, m(x1) - slack)
    cy1 = max(0.0, m(y1) - slack)
    cx2 = min(1.0, m(x2) + slack)
    cy2 = min(1.0, m(y2) + slack)
    return (
        int(cx1 * width),
        int(cy1 * height),
        max(int(cx2 * width), int(cx1 * width) + 1),
        max(int(cy2 * height), int(cy1 * height) + 1),
    )


def _ocr_zone(
    image: np.ndarray,
    engines: list[OcrEngine],
    *,
    min_width: int = 900,
) -> tuple[str, float, str]:
    if image.size == 0:
        return "", 0.0, ""
    h, w = image.shape[:2]
    scale = max(1.0, float(min_width) / float(max(w, 1)))
    if scale > 1.0:
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Plain grayscale reads big smooth digits best; Otsu helps low-contrast
    # red-on-red. Two variants is the speed/quality sweet spot (CLAHE rarely
    # beat both and tripled cost).
    variants = [
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.COLOR_GRAY2BGR,
        ),
    ]
    best_text, best_conf, best_engine = "", 0.0, ""
    for engine in engines:
        for prepared in variants:
            try:
                text, confidence = engine.recognize(prepared)
            except Exception:  # noqa: BLE001
                continue
            if _text_score(text=text, confidence=confidence) > _text_score(
                text=best_text, confidence=best_conf
            ):
                best_text, best_conf, best_engine = (
                    _normalize_ocr_text(text),
                    confidence,
                    engine.name,
                )
            # Good-enough early exit skips the slow fallback engine entirely.
            if best_conf >= 0.65 and len(best_text) >= 2:
                return best_text, best_conf, best_engine
    return best_text, best_conf, best_engine


def _recognize_zonal(
    crop: np.ndarray,
    *,
    engines: list[OcrEngine],
) -> _ZonalResult:
    if not engines or crop.size == 0:
        return _ZonalResult({}, "", 0.0, "")
    h, w = crop.shape[:2]
    zone_texts: dict[str, str] = {}
    confidences: list[float] = []
    engine_name = ""
    # Cyrillic zones can't use the (broken) Paddle-ru model; Tesseract rus
    # handles the few text fields. Numeric zones use the working Paddle-en.
    cyrillic_zones = {"product_name", "additional_info"}
    for zone, frac in _TAG_ZONES.items():
        x1, y1, x2, y2 = _zone_box(frac, w, h)
        sub = crop[y1:y2, x1:x2]
        if sub.size == 0:
            continue
        if zone in cyrillic_zones:
            text = _cyrillic_ocr(sub)
            conf, eng = (0.5 if text else 0.0), "tesseract_rus"
        else:
            text, conf, eng = _ocr_zone(sub, engines)
        if text:
            zone_texts[zone] = text
            confidences.append(conf)
            engine_name = engine_name or eng
    combined = "\n".join(zone_texts.values())
    return _ZonalResult(zone_texts, combined, _mean(confidences), engine_name)


def _cyrillic_ocr(image: np.ndarray) -> str:
    """Russian text zones via Tesseract (Paddle-ru is broken here)."""
    try:
        import pytesseract
    except Exception:  # noqa: BLE001
        return ""
    h, w = image.shape[:2]
    scale = max(1.0, 1100.0 / float(max(w, 1)))
    if scale > 1.0:
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    best = ""
    for img in (
        gray,
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ):
        try:
            text = pytesseract.image_to_string(
                img, lang="rus+eng", config="--oem 1 --psm 6"
            )
        except Exception:  # noqa: BLE001
            continue
        text = _normalize_ocr_text(text)
        if len(text) > len(best):
            best = text
    return best


_KOPECK_PRIORITY = ("99", "90", "50", "49", "00", "95", "98", "49", "59", "79")


def _best_price(text: str) -> str:
    """Reassemble ``rubles.kopecks``.

    On these tags the price is a big integer (rubles) with a small
    superscript 2-digit kopecks, so OCR returns them as separate tokens
    (e.g. ``1199`` and ``99``) in arbitrary order. Combine them instead of
    dropping the kopecks (the metric needs exact price within 0.01).
    """
    direct = _extract_prices(text)
    if direct:
        return max(direct, key=_price_sort_key)
    tokens = re.findall(r"\d+", text)
    ints = [t for t in tokens if 2 <= len(t) <= 6]
    if not ints:
        return ""
    big = [t for t in ints if len(t) >= 3]
    if not big:
        return f"{int(max(ints, key=int))}.00"
    max_len = max(len(t) for t in big)
    rub = max((t for t in big if len(t) == max_len), key=lambda s: int(s))
    kop = ""
    if rub in tokens:
        i = tokens.index(rub)
        for j in (i + 1, i - 1):
            if 0 <= j < len(tokens) and len(tokens[j]) == 2:
                kop = tokens[j]
                break
    if not kop:
        two = [t for t in ints if len(t) == 2]
        for pref in _KOPECK_PRIORITY:
            if pref in two:
                kop = pref
                break
        if not kop and two:
            kop = two[0]
    return f"{int(rub)}.{kop or '00'}"


def _best_discount(text: str) -> str:
    m = DISCOUNT_RE.search(text)
    if m:
        return re.sub(r"\s+", "", m.group(0)).replace("−", "-")
    # OCR often drops the leading minus; a discount badge is always negative.
    m = re.search(r"(?<!\d)(\d{1,2})\s*%", text)
    if m:
        return f"-{m.group(1)}%"
    return ""


def _fields_from_zones(zone_texts: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    name = _extract_product_name(zone_texts.get("product_name", ""))
    if name:
        fields["product_name"] = name

    def _plausible_price(v: str) -> bool:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        # retail tag prices: ~10..100000 rub; rejects volumes (0.75) & junk.
        return 10.0 <= f <= 100000.0

    pd = _best_price(zone_texts.get("price_default", ""))
    pc = _best_price(zone_texts.get("price_card", ""))
    pd = pd if _plausible_price(pd) else ""
    pc = pc if _plausible_price(pc) else ""
    if pc:
        fields["price_card"] = pc
    # price_default only if it's a distinct plausible price (not a dup of the
    # card price, which means the zone just bled the big number).
    if pd and pd != pc:
        fields["price_default"] = pd

    disc = _best_discount(
        zone_texts.get("discount", "") + " " + zone_texts.get("price_card", "")
    )
    if disc:
        fields["discount_amount"] = disc

    # Only emit barcode/sku when they validate; a wrong value scores the same
    # as empty AND a garbage barcode can mis-key GT matching (task spec 6).
    price_digits = "".join(
        re.sub(r"\D", "", v) for v in (pd, pc, disc) if v
    )

    def _is_price_concat(s: str) -> bool:
        return bool(price_digits) and (s in price_digits or price_digits.startswith(s[:6]))

    barcode_src = zone_texts.get("barcode", "") + " " + zone_texts.get("sku_date", "")
    ean = _first_valid_ean13(barcode_src)
    if ean and not _is_price_concat(ean):
        fields["barcode"] = ean

    sku = _extract_sku(zone_texts.get("sku_date", ""), exclude=fields.get("barcode", ""))
    if sku and len(sku) >= 8 and not _is_price_concat(sku):
        fields["id_sku"] = sku
    date = DATE_RE.search(zone_texts.get("sku_date", ""))
    if date:
        fields["print_datetime"] = date.group(0)

    symbol = _extract_special_symbol(zone_texts.get("special", ""))
    if symbol:
        fields["special_symbols"] = symbol
    info = _extract_additional_info(
        zone_texts.get("additional_info", "")
        + "\n"
        + zone_texts.get("product_name", "")
    )
    if info:
        fields["additional_info"] = info
    return fields


def parse_ocr_fields(text: str) -> dict[str, str]:
    normalized = _normalize_ocr_text(text)
    if not normalized:
        return {}

    fields: dict[str, str] = {}
    prices = _extract_prices(normalized)
    if prices:
        unique_prices = list(dict.fromkeys(prices))
        if len(unique_prices) == 1:
            fields["price_card"] = unique_prices[0]
        else:
            numeric_prices = sorted(unique_prices, key=_price_sort_key, reverse=True)
            fields["price_default"] = numeric_prices[0]
            fields["price_card"] = numeric_prices[-1]

    discount = DISCOUNT_RE.search(normalized)
    if discount:
        fields["discount_amount"] = re.sub(r"\s+", "", discount.group(0)).replace("−", "-")

    date = DATE_RE.search(normalized)
    if date:
        fields["print_datetime"] = date.group(0)

    ean13 = _first_valid_ean13(normalized)
    if ean13:
        fields["barcode"] = ean13

    sku = _extract_sku(normalized, exclude=ean13)
    if sku:
        fields["id_sku"] = sku

    product_name = _extract_product_name(normalized)
    if product_name:
        fields["product_name"] = product_name

    additional_info = _extract_additional_info(normalized)
    if additional_info:
        fields["additional_info"] = additional_info

    symbol = _extract_special_symbol(normalized)
    if symbol:
        fields["special_symbols"] = symbol

    return fields


def classify_tag_color(crop: np.ndarray) -> str:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 45, 70), (10, 255, 255)),
        cv2.inRange(hsv, (170, 45, 70), (180, 255, 255)),
    )
    yellow_mask = cv2.inRange(hsv, (14, 55, 85), (40, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 40, 55), (90, 255, 255))
    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "yellow": int(cv2.countNonZero(yellow_mask)),
        "green": int(cv2.countNonZero(green_mask)),
    }
    color, count = max(counts.items(), key=lambda item: item[1])
    area = max(crop.shape[0] * crop.shape[1], 1)
    if count / float(area) < 0.015:
        return ""
    return color


def _orientation_variants(crop: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "none": crop,
        "rotate_90_cw": cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
        "rotate_90_ccw": cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "rotate_180": cv2.rotate(crop, cv2.ROTATE_180),
    }


def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max(1.0, 900.0 / float(max(width, 1)))
    if scale > 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _build_frame_lookup(
    sampled_frames: list[SampledFrameMetadata],
) -> dict[int, SampledFrameMetadata]:
    return {frame.frame_index: frame for frame in sampled_frames}


def _load_crop_image(
    *,
    crop: CropCandidate,
    frame_lookup: dict[int, SampledFrameMetadata],
) -> np.ndarray | None:
    frame_meta = frame_lookup.get(crop.frame_index)
    if frame_meta is None or frame_meta.local_frame_path is None:
        return None

    frame_path = Path(frame_meta.local_frame_path)
    if not frame_path.exists():
        return None

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None

    padded_bbox = clip_bbox_to_frame(
        crop.padded_bbox,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )
    if padded_bbox is None:
        return None

    crop_image = frame[
        padded_bbox.y_min : padded_bbox.y_max,
        padded_bbox.x_min : padded_bbox.x_max,
    ]
    if crop_image.size == 0:
        return None
    return crop_image


def _extract_prices(text: str) -> list[str]:
    prices: list[str] = []
    for match in PRICE_RE.finditer(text):
        integer_part = re.sub(r"\D+", "", match.group(1))
        fraction_part = re.sub(r"\D+", "", match.group(2))
        if not integer_part or len(fraction_part) != 2:
            continue
        prices.append(f"{int(integer_part)}.{fraction_part}")
    return prices


def _price_sort_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _first_valid_ean13(text: str) -> str:
    for match in EAN13_RE.finditer(text):
        digits = re.sub(r"\D+", "", match.group(0))
        if is_valid_ean13(digits):
            return digits
    return ""


def _extract_sku(text: str, *, exclude: str) -> str:
    # GT id_sku is exactly 12 digits. Shorter runs at this resolution are
    # almost always price/date fragments, so emitting them is pure downside
    # (a wrong value scores like empty). Require an exact 12-digit token.
    for match in re.finditer(r"(?<!\d)\d{12}(?!\d)", re.sub(r"[\s-]", "", text)):
        digits = match.group(0)
        if digits != exclude:
            return digits
    return ""


def is_valid_ean13(value: str) -> bool:
    if not re.fullmatch(r"\d{13}", value):
        return False
    digits = [int(char) for char in value]
    checksum = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return checksum == digits[-1]


def _extract_product_name(text: str) -> str:
    lines = _clean_lines(text)
    candidates: list[str] = []
    for line in lines:
        folded = line.casefold()
        if any(stopword in folded for stopword in PRODUCT_STOPWORDS):
            continue
        if sum(char.isdigit() for char in line) > max(len(line) // 3, 2):
            continue
        if len(line) < 4:
            continue
        candidates.append(line)
        if len(candidates) >= 3:
            break
    return " ".join(candidates).strip()


def _extract_additional_info(text: str) -> str:
    for line in _clean_lines(text):
        folded = line.casefold()
        if any(
            keyword in folded
            for keyword in ("сухое", "полусухое", "полусладкое", "сладкое", "брют")
        ):
            return line
    return ""


def _extract_special_symbol(text: str) -> str:
    for token in re.findall(r"\b[КШЛK]\b", text, flags=re.IGNORECASE):
        return "К" if token.upper() == "K" else token.upper()
    return ""


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip(" .,:;|")
        for line in text.splitlines()
        if re.sub(r"\s+", " ", line).strip(" .,:;|")
    ]


def _normalize_ocr_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace(MOJIBAKE_NO_VALUE, NO_VALUE)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _text_score(*, text: str, confidence: float) -> float:
    normalized = _normalize_ocr_text(text)
    if not normalized:
        return 0.0
    digit_bonus = min(sum(char.isdigit() for char in normalized) / 20.0, 1.0)
    price_bonus = 0.4 if PRICE_RE.search(normalized) else 0.0
    cyrillic_bonus = 0.2 if re.search(r"[А-Яа-я]", normalized) else 0.0
    return confidence + digit_bonus + price_bonus + cyrillic_bonus + min(len(normalized) / 120.0, 0.5)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    field_name: str,
    warnings: list[str],
) -> int:
    try:
        coerced = int(default if value is None else value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced <= 0:
        warnings.append(f"{field_name} must be > 0. Falling back to {default}.")
        return default
    return coerced


def _coerce_ratio_inclusive_zero(
    value: Any,
    *,
    default: float,
    field_name: str,
    warnings: list[str],
) -> float:
    try:
        coerced = float(default if value is None else value)
    except (TypeError, ValueError):
        warnings.append(f"Invalid {field_name} '{value}'. Falling back to {default}.")
        return default
    if coerced < 0 or coerced > 1:
        warnings.append(f"{field_name} must be within [0, 1]. Falling back to {default}.")
        return default
    return coerced
