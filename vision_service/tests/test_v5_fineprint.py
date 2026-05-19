"""Unit coverage for the v5 fine-print extractor (id_sku/datetime/code).

These three fields gate the metric (oracle: recovering them lifts
26_12-20 score 0.014 -> 0.96). Consensus must reconstruct the exact SKU
from several noisy reads of the same static value.
"""
from __future__ import annotations

from app.pipelines.price_tag_v5.stages.block_ocr_catalog import (
    V5BlockOcrCatalogStage as S,
)


def test_sku_candidates_extracts_12_digits() -> None:
    c = S._sku_candidates(["270102701074", "x 370204501518 y", "12 34"])
    assert "270102701074" in c
    assert "370204501518" in c


def test_sku_candidates_window_with_prefix() -> None:
    # 13-digit noisy read -> the (27|37)-prefixed 12-window is kept
    c = S._sku_candidates(["2701027010749"])
    assert "270102701074" in c


def test_consensus_sku_majority_value() -> None:
    cands = ["270102701074", "270102701074", "270102701999"]
    assert S._consensus_sku(cands) == "270102701074"


def test_consensus_sku_per_position_recovers_exact() -> None:
    # no single read is correct; per-position majority reconstructs it
    cands = ["270102701074", "270102701084", "270102701075",
             "270102701074", "270902701074"]
    assert S._consensus_sku(cands) == "270102701074"


def test_consensus_sku_empty() -> None:
    assert S._consensus_sku([]) == ""
    assert S._consensus_sku(["123"]) == ""


def test_date_candidates() -> None:
    c = S._date_candidates(["напечатано 17.02.2026 1:14 ", "03,04,2026"])
    assert "17.02.2026 1:14" in c
    assert "03.04.2026" in c


def test_code_candidates_patterns() -> None:
    assert "01_026015 - 026016" in S._code_candidates(["x 01_026015 - 026016"])
    assert "13_043015" in S._code_candidates(["13_043015"])
    assert any("026005" in c for c in S._code_candidates(["026005 - 026013"]))


def test_mode_picks_most_common() -> None:
    assert S._mode(["a", "b", "a"]) == "a"
    assert S._mode([]) == ""
