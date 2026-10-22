"""ESCI product corpus (all US small-set products) and the text variants used for indexing."""

import pandas as pd

from cartrank.data import esci
from cartrank.data.paths import PROCESSED

CORPUS_PATH = PROCESSED / "esci" / "corpus.parquet"
BULLET_CHARS = 500
DESC_CHARS = 300

# Ablation variants (Phase 1.1); `tbcb` is the default unless the ablation says otherwise.
VARIANTS = ("title", "tbc", "tbcb", "tbcbd")
DEFAULT_VARIANT = "tbcb"


def _clean(s: str | None) -> str:
    return " ".join(s.split()) if isinstance(s, str) else ""


def product_text(row: pd.Series, variant: str = DEFAULT_VARIANT) -> str:
    parts = [_clean(row.product_title)]
    if variant != "title":
        parts += [_clean(row.product_brand), _clean(row.product_color)]
    if variant in ("tbcb", "tbcbd"):
        parts.append(_clean(row.product_bullet_point)[:BULLET_CHARS])
    if variant == "tbcbd":
        parts.append(_clean(row.product_description)[:DESC_CHARS])
    return ". ".join(p for p in parts if p)


def build_corpus() -> pd.DataFrame:
    ex = esci.load_examples()
    prod = esci.load_products(ex.product_id.unique()).drop_duplicates("product_id")
    prod = prod.sort_values("product_id").reset_index(drop=True)
    for v in VARIANTS:
        prod[f"text_{v}"] = [product_text(r, v) for r in prod.itertuples()]
    return prod[["product_id", *[f"text_{v}" for v in VARIANTS]]]


def load_corpus() -> pd.DataFrame:
    if not CORPUS_PATH.exists():
        build_corpus().to_parquet(CORPUS_PATH)
    return pd.read_parquet(CORPUS_PATH)
