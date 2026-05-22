"""Unit coverage for the v5 CatalogResolver invariants.

Locks the measured behaviour: garbled-but-distinctive brand tokens still
resolve (proportional fuzzy), a confidently accepted name now ALWAYS emits
a barcode, and low-evidence / generic queries stay withheld (precision is
the thing that protects the GT match key).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipelines.price_tag_v5.catalog import (
    CatalogResolver,
    _editdist,
    _trigrams,
    brand_tokens,
    ean13_ok,
    to_ean13,
)

_ROWS = [
    ("Вино LE VIGNE DI SAMMARCO NEGROAMARO IGP кр. п/ сух. (Италия) 0.75L",
     "8051070512049"),
    ("Вино LE VIGNE DI SAMMARCO NEGROAMARO IGP кр. п/ сух. (Италия) 0.75L",
     "8051070512131"),
    ("Вино POGGIO TOSCO Россо Тоскано ИГТ сорт. кр. сух. (Италия) 0.75L",
     "8006600103006"),
    ("Вино CITRAN Бордо Суперьор кр. сух. (Франция) 0.75L", "4690491122587"),
]


@pytest.fixture(scope="module")
def resolver(tmp_path_factory: pytest.TempPathFactory) -> CatalogResolver:
    csv_path: Path = tmp_path_factory.mktemp("cat") / "db.csv"
    lines = ["fullname;code"] + [f"{n};{c}" for n, c in _ROWS]
    csv_path.write_bytes("\n".join(lines).encode("cp1251"))
    return CatalogResolver(csv_path)


def test_ean13_ok_and_to_ean13() -> None:
    assert ean13_ok("8051070512049")
    assert not ean13_ok("8051070512040")
    assert to_ean13("08051070512049") == "8051070512049"  # UPC/0-pad strip
    assert to_ean13("4690491122587") == "4690491122587"
    assert to_ean13("abc") == ""


def test_trigrams_and_editdist() -> None:
    assert "SAM" in _trigrams("SAMMARCO")
    assert _editdist("SAMMARCO", "SANNARCO") == 2
    assert _editdist("POGGIO", "POGGIO") == 0
    assert _editdist("AB", "ABCDEFGHIJ") == 99  # length-pruned


def test_brand_tokens_strips_garbled_vino_prefix() -> None:
    toks = brand_tokens(["BPHOLEVIGNED", "SAMARCO", "NEGROAMARO IGS"])
    assert "SAMARCO" in toks
    assert "BPHO" not in toks


def test_accepted_implies_name_and_barcode(resolver: CatalogResolver) -> None:
    # Threshold-independent invariant of the new emit policy: whenever a
    # match is accepted it MUST carry both a product_name and a barcode
    # (the old code withheld barcode on ambiguity -> 0 recall).
    for q in (["BAHOLEVIGNEDI", "SANNARCO", "NEGROAMARD"], ["CITRAN"],
              ["POGGIO"], ["XQZJ"], ["СУХОЕ"], ["LE", "VIGNE"]):
        m = resolver.resolve(q, category="all")
        if m is not None and m.accepted:
            assert m.product_name
            assert m.barcode
            assert m.barcode in m.candidate_codes


_REAL_DB = Path("data/db_hack.csv")


@pytest.mark.skipif(not _REAL_DB.exists(),
                    reason="real db_hack.csv not linked into worktree")
def test_real_catalog_garbled_brand_resolves_with_barcode() -> None:
    # Mirrors the measured catalog probe: SANNARCO (edit-dist 2 from
    # SAMMARCO) resolves to the right wine AND emits the GT barcode.
    r = CatalogResolver(_REAL_DB)
    m = r.resolve(["BAHOLEVIGNEDI", "SANNARCO", "NEGROAMARD"],
                  category="wine")
    assert m is not None and m.accepted
    assert "SAMMARCO" in m.product_name.upper()
    assert m.barcode == "8051070512049"  # sorted-first == GT here


def test_generic_or_garbage_tokens_withheld(resolver: CatalogResolver) -> None:
    # pure descriptors / random junk must NOT produce an identity
    for q in (["СУХОЕ", "КР"], ["XQZJ", "ZZZZ"], ["AB", "CD"]):
        m = resolver.resolve(q, category="all")
        assert m is None or (not m.accepted and m.barcode == "")


def test_not_accepted_has_no_name_or_barcode(
    resolver: CatalogResolver,
) -> None:
    m = resolver.resolve(["NEGROAMARD"], category="all")  # only generic-ish
    if m is not None and not m.accepted:
        assert m.product_name == ""
        assert m.barcode == ""


def test_best_effort_name_is_catalog_guess_not_product_name(
    resolver: CatalogResolver,
) -> None:
    m = resolver.resolve(["NEGROAMARD"], category="all", best_effort=True)

    assert m is not None
    assert not m.accepted
    assert m.product_name == ""
    assert m.barcode == ""
    assert m.catalog_guess_name
    assert m.status == "catalog_guess"
