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
            for i, ts in enumerate(toks_per):
                for t in ts:
                    inv[t].append(i)
            self._idx[bucket] = {
                "names": names, "toks": toks_per, "idf": idf, "inv": inv,
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
    ) -> CatalogMatch | None:
        qtoks = [t for t in brand_tokens(block_texts)
                 if t not in _STOP_TOKENS]
        if not qtoks:
            return None
        idx = self._idx.get(category) or self._idx["all"]
        idf, inv, names, toks_per = (
            idx["idf"], idx["inv"], idx["names"], idx["toks"]
        )

        def qmatch(q: str, nameset: set[str]) -> tuple[bool, str]:
            if q in nameset:
                return True, q
            for nt in nameset:
                if len(q) >= 4 and (q in nt or nt in q):
                    return True, nt
                if len(q) >= 4 and abs(len(q) - len(nt)) <= 1 and _editdist(q, nt) <= 1:
                    return True, nt
            return False, ""

        # candidate names = those sharing any exact query token (fast path);
        # fall back to a fuzzy scan only if the fast path is empty.
        cand: set[int] = set()
        for q in qtoks:
            cand.update(inv.get(q, ()))
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
        # Barcode is the GT match key — only commit it when it is NOT a
        # coin-flip: a single EAN candidate, OR a digit hint that confirms
        # one candidate (edit-distance<=1 + checksum). Otherwise keep the
        # (trusted) name but leave barcode empty rather than mis-key GT.
        barcode, cands = self._pick_barcode(best_name, barcode_hint)
        hint = re.sub(r"\D", "", barcode_hint or "")
        if not name_ok:
            barcode = ""
        elif len(cands) > 1:
            if hint and barcode and _editdist(barcode, hint) <= 1 and ean13_ok(barcode):
                pass  # digit-confirmed
            else:
                barcode = ""  # ambiguous variant — don't guess the key
        return CatalogMatch(
            product_name=best_name if name_ok else "",
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
