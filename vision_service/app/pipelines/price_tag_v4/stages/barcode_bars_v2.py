from __future__ import annotations

from app.pipelines.price_tag_v3.stages.barcode_bars_decode import BarcodeBarsDecodeStage


class BarcodeBarsV2Stage(BarcodeBarsDecodeStage):
    """v4 wrapper for the measured barcode-bars path.

    The local benchmark config keeps this disabled while assignment coverage is
    being tuned. The separate stage name lets us measure future rectification
    changes independently from v3 without mutating the v3 implementation.
    """

    name = "BarcodeBarsV2Stage"
