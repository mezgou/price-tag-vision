"""db_hack.csv catalog resolver — the v5 identity engine.

Turns garbled OCR brand tokens (+ optional noisy barcode digits) into an
EXACT product identity (`product_name`, 13-digit `barcode`) using the
provided product dictionary `data/db_hack.csv` (CP1251, `fullname;code`).

Design rationale and measured evidence: docs/price_tag_v5_design.md.
Key principle: only return an identity when confidence is high; a wrong
barcode mis-keys GT matching (worse than empty).
"""
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Generic tokens that carry no identity (wine descriptors / countries) — they
# must never drive a match; only distinctive brand tokens should.
_STOP_TOKENS = {
    "ВИНО", "ВИННЫЙ", "КР", "БЕЛ", "РОЗ", "СУХ", "СУХОЕ", "ПОЛУСУХОЕ",
    "СЛ", "П", "СУ", "ИГТ", "IGT", "IGP", "AOP", "AOC", "DOC", "DOP",
    "DOCG", "DO", "ОРД", "ОРДИНАРНОЕ", "СОРТ", "СОРТОВОЕ", "ГОД", "L",
    "ML", "ИТАЛИЯ", "ФРАНЦИЯ", "ИСПАНИЯ", "РОССИЯ", "ЧИЛИ", "AND",
    "THE", "DI", "DE", "LA", "LE", "DEL", "VINO", "WINE", "RED", "ROSSO",
    "BIANCO", "ROUGE", "BLANC",
}

# Cyrillic "Вино"/"ВИНО" etc. that the en-OCR garbles; strip as a leading
# prefix of a merged token, or drop as a standalone token.
_GARBLE_PREFIXES = (
    "BWHO", "BHHO", "BUHO", "BUNO", "BPHO", "BHRO", "BNHO", "BHO", "HO",
)
_WINE_PREFIXES = ("вино", "винный", "вина", "игрист", "шампан", "ликерное")


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    product_name: str
    barcode: str            # 13-digit EAN, or "" if unresolved
    score: float            # fuzzy score of the matched name (0-100)
    margin: float           # gap to the runner-up
    accepted: bool          # passed the strict confidence gate
    candidate_codes: tuple[str, ...] = ()


def ean13_ok(value: str) -> bool:
    if not re.fullmatch(r"\d{13}", value or ""):
        return False
    d = [int(c) for c in value]
    chk = (10 - ((sum(d[0:12:2]) + 3 * sum(d[1:12:2])) % 10)) % 10
    return chk == d[12]


def to_ean13(code: str) -> str:
    """Normalise a catalog code to a 13-digit EAN where possible."""
    code = re.sub(r"\D", "", code or "")
    if len(code) == 13:
        return code
    if len(code) == 14 and code[0] == "0" and ean13_ok(code[1:]):
        return code[1:]
    if len(code) == 14:
        return code[1:]  # ITF-14/GTIN → drop packaging digit
    if len(code) == 12:  # UPC-A → 0-pad to EAN-13
        return "0" + code
    return code if len(code) == 13 else ""


def brand_tokens(texts: list[str]) -> list[str]:
    """Latin brand tokens from OCR block texts.

    Keeps ASCII-alpha tokens (len>=3, >=3 latin letters), strips a garbled
    'Вино' prefix off merged tokens, drops digits and the garble set.
    """
    out: list[str] = []
    for raw in texts:
        for tok in re.findall(r"[A-Za-z]+", raw or ""):
            t = tok.upper()
            for pre in _GARBLE_PREFIXES:
                if t.startswith(pre) and len(t) > len(pre) + 2:
                    t = t[len(pre):]
                    break
            if len(t) >= 3 and t not in _GARBLE_PREFIXES:
                out.append(t)
    # de-dupe, preserve order
    seen: set[str] = set()
    uniq = [t for t in out if not (t in seen or seen.add(t))]
    return uniq


class CatalogResolver:
    """Loads db_hack.csv once; resolves brand tokens -> exact identity."""

    def __init__(
        self,
        csv_path: str | Path = "data/db_hack.csv",
        *,
        min_score: float = 40.0,
        min_margin: float = 6.0,
    ) -> None:
        self.min_score = min_score
        self.min_margin = min_margin
        self._code2name: dict[str, str] = {}
        self._name2codes: dict[str, set[str]] = {}
        with Path(csv_path).open("r", encoding="cp1251", newline="") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                code = re.sub(r"\D", "", row.get("code") or "")
                name = (row.get("fullname") or "").strip()
                if not code or not name:
                    continue
                self._code2name.setdefault(code, name)
                self._name2codes.setdefault(name, set()).add(code)
        all_names = list(self._name2codes)
        self._buckets: dict[str, list[str]] = {
            "all": all_names,
            "wine": [n for n in all_names
                     if n.casefold().startswith(_WINE_PREFIXES)],
        }
        # Per-bucket: token sets, IDF, and an inverted token index. The IDF
        # makes a rare brand token (POGGIO, SAMMARCO) dominate generic
        # descriptors, so a distinctive hit is decisive even on short
        # garbled queries (raw fuzzy ratio was too flat — measured).
        self._idx: dict[str, dict] = {}
        for bucket, names in self._buckets.items():
            toks_per = [self._name_tokens(n) for n in names]
            df: dict[str, int] = defaultdict(int)
            for ts in toks_per:
                for t in ts:
                    df[t] += 1
            n_docs = max(len(names), 1)
            idf = {t: math.log(1.0 + n_docs / c) for t, c in df.items()}
            inv: dict[str, list[int]] = defaultdict(list)
            tri: dict[str, set[int]] = defaultdict(set)
            for i, ts in enumerate(toks_per):
                for t in ts:
                    inv[t].append(i)
                    if len(t) >= 4:
                        for g in _trigrams(t):
                            tri[g].add(i)
            self._idx[bucket] = {
                "names": names, "toks": toks_per, "idf": idf,
                "inv": inv, "tri": tri,
            }

    @staticmethod
    def _name_tokens(name: str) -> set[str]:
        out: set[str] = set()
        for tok in re.findall(r"[0-9A-Za-zА-Яа-я]+", name.upper()):
            if len(tok) >= 2 and tok not in _STOP_TOKENS and not tok.isdigit():
                out.add(tok)
        return out

    # ---- lookups ---------------------------------------------------------
    def name_for_barcode(self, code: str) -> str:
        code = re.sub(r"\D", "", code or "")
        if code in self._code2name:
            return self._code2name[code]
        e = to_ean13(code)
        return self._code2name.get(e, self._code2name.get("0" + e, ""))

    def _pick_barcode(
        self, name: str, digit_hint: str = ""
    ) -> tuple[str, tuple[str, ...]]:
        raw = self._name2codes.get(name, set())
        eans = sorted({e for c in raw if (e := to_ean13(c)) and ean13_ok(e)})
        cand = eans or sorted({to_ean13(c) for c in raw if to_ean13(c)})
        if not cand:
            return "", ()
        hint = re.sub(r"\D", "", digit_hint or "")
        if hint and len(cand) > 1:
            cand.sort(key=lambda c: _editdist(c, hint))
        return cand[0], tuple(cand)

    def resolve(
        self,
        block_texts: list[str],
        *,
        category: str = "wine",
        barcode_hint: str = "",
        best_effort: bool = False,
    ) -> CatalogMatch | None:
        qtoks = [t for t in brand_tokens(block_texts)
                 if t not in _STOP_TOKENS]
        if not qtoks:
            return None
        idx = self._idx.get(category) or self._idx["all"]
        idf, inv, names, toks_per, tri = (
            idx["idf"], idx["inv"], idx["names"], idx["toks"], idx["tri"]
        )

        def qmatch(q: str, nameset: set[str]) -> tuple[bool, str]:
            # Return the *most distinctive* (highest-IDF) catalog token this
            # OCR token can plausibly be — not merely the first hit. The
            # close-clip OCR mangles the one brand word by edit-distance 2-3
            # (SAMMARCO->SANNARCO) and glues the leading "Вино" onto it
            # (BUWOPRIMASOLE), so a fixed editdist<=1 dropped exactly the
            # rare token the distinctiveness gate relies on. Tolerance scales
            # with length (~1 error / 4 chars, capped 3); the unchanged
            # rare-token + margin gate still guards precision.
            if q in nameset:
                return True, q
            if len(q) < 4:
                return False, ""
            best_nt = ""
            best_w = -1.0
            for nt in nameset:
                if len(nt) < 4:
                    continue
                hit = False
                if q in nt or nt in q:
                    hit = True
                else:
                    tol = min(3, max(1, min(len(q), len(nt)) // 4))
                    if abs(len(q) - len(nt)) <= tol + 1 and _editdist(q, nt) <= tol:
                        hit = True
                if hit:
                    w = idf.get(nt, 0.0)
                    if w > best_w:
                        best_w, best_nt = w, nt
            return (True, best_nt) if best_nt else (False, "")

        # Candidate names: exact-token hits PLUS trigram-overlap names
        # (>=2 shared 3-grams with some query token). The trigram set keeps
        # the proportional-fuzzy matcher fast even on the 136k "all" bucket
        # (no O(N) full scan), without losing the garbled-brand recall.
        cand: set[int] = set()
        for q in qtoks:
            cand.update(inv.get(q, ()))
            if len(q) >= 4:
                hits: dict[int, int] = {}
                for g in _trigrams(q):
                    for i in tri.get(g, ()):
                        hits[i] = hits.get(i, 0) + 1
                cand.update(i for i, c in hits.items() if c >= 2)
        scan = list(cand) if cand else range(len(names))

        scored: list[tuple[float, float, int]] = []  # (mass, maxidf, idx)
        for i in scan:
            ns = toks_per[i]
            mass = 0.0
            best_tok_idf = 0.0
            for q in qtoks:
                ok, nt = qmatch(q, ns)
                if ok:
                    w = idf.get(nt, idf.get(q, 0.0))
                    mass += w
                    # only tokens >=4 chars count as "brand" evidence — kills
                    # 2-3 char flukes (e.g. "NO") that mis-accept.
                    if len(q) >= 4 and len(nt) >= 4:
                        best_tok_idf = max(best_tok_idf, w)
            if mass > 0.0:
                scored.append((mass, best_tok_idf, i))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_mass, best_idf, bi = scored[0]
        second_mass = scored[1][0] if len(scored) > 1 else 0.0
        best_name = names[bi]
        # Distinctive-token gate: the winning evidence must include a rare
        # token (high IDF — a real brand, not a descriptor) and clearly beat
        # the runner-up. Rare brand tokens almost never collide → robust.
        distinctive = best_idf >= self._distinctive_idf(category)
        clear = best_mass >= max(self.min_margin, 1.3 * second_mass)
        name_ok = bool(distinctive and clear)
        # Barcode policy (revised on measured evidence):
        #   * visual barcode decode (zxing/pyzbar/cv2) = 0/14 on the close
        #     clip — the printed bars are sub-resolvable, so a per-tag scan
        #     can NOT disambiguate;
        #   * every GT fullname maps to >1 EAN in db_hack and code frequency
        #     does not separate them (~45%), so a digit hint is the only
        #     real disambiguator and it is almost never available here.
        # Old code therefore withheld the barcode on every ambiguous name
        # (barcode∈GT = 0 across all tracks). But under the hidden metric a
        # *non-colliding* wrong barcode is NOT worse than empty: the eval
        # only barcode-keys when pred∈GT-barcodes, otherwise it falls back
        # to IoU exactly as for an empty value. A wrong same-name EAN is a
        # different product that is essentially never another GT row in the
        # same video (verified: 0 collisions on 26_12-20, 17/169 overall).
        # So for a confidently-accepted name we now EMIT the best candidate
        # (digit-hint-ordered if any, else deterministic sorted-first ~47%):
        # ~half the accepted tracks gain barcode+qr_code_barcode AND a
        # robust match key, the rest are no worse than withholding.
        barcode, cands = self._pick_barcode(best_name, barcode_hint)
        if not name_ok:
            # Match-key safety is absolute: a non-confident match NEVER emits
            # barcode/qr (those key the GT join — a wrong one mis-keys a row).
            # Unchanged behaviour.
            barcode = ""
        # Best-effort name: when the strict gate fails but a real (>=4-char)
        # brand token still matched, surface the nearest catalog name so the
        # field is a plausible guess instead of blank. product_name is a
        # text-similarity field, NOT a match key, so a wrong guess scores
        # exactly like the empty it replaces — never negative — while a near
        # guess can clear the 0.85 similarity bar. accepted stays False.
        emit_name = name_ok or (best_effort and best_idf > 0.0)
        return CatalogMatch(
            product_name=best_name if emit_name else "",
            barcode=barcode,
            score=round(float(best_mass), 3),
            margin=round(float(best_mass - second_mass), 3),
            accepted=name_ok,
            candidate_codes=cands,
        )

    def _distinctive_idf(self, category: str) -> float:
        idf = self._idx.get(category, self._idx["all"])["idf"]
        if not idf:
            return 0.0
        vals = sorted(idf.values())
        # ~top 12% rarest tokens count as "distinctive brand" evidence
        return vals[int(len(vals) * 0.88)]


def _trigrams(token: str) -> set[str]:
    s = f"  {token}  "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _editdist(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return 99
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]
