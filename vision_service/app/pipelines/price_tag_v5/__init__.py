"""price_tag_v5 — catalog-anchored recognition (GT-free).

Identity from brand-name fuzzy match into data/db_hack.csv; prices from one
fast full-crop OCR pass. See docs/price_tag_v5_design.md.
"""

# catalog first so the submodule is in sys.modules before pipeline/stages
# import it (avoids a partial-package circular import).
from app.pipelines.price_tag_v5.catalog import CatalogMatch, CatalogResolver
from app.pipelines.price_tag_v5.pipeline import PriceTagV5Pipeline

__all__ = ["CatalogResolver", "CatalogMatch", "PriceTagV5Pipeline"]
