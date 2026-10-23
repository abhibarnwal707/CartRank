"""Demo catalog: Office_Products items with a usable price, plus the text used for indexing."""

import pandas as pd

from cartrank.data.paths import PROCESSED
from cartrank.search.corpus import BULLET_CHARS

CATALOG_PATH = PROCESSED / "catalog.parquet"


def _clean(s: str | None) -> str:
    return " ".join(s.split()) if isinstance(s, str) else ""


def catalog_text(title: str, store: str | None, features: list[str]) -> str:
    """Same layout as the ESCI product text (title, brand, bullets); colour is not in this data."""
    parts = [_clean(title), _clean(store), _clean(" ".join(features))[:BULLET_CHARS]]
    return ". ".join(p for p in parts if p)


def build_catalog(meta: pd.DataFrame) -> pd.DataFrame:
    """Keep items with a positive price and a non-empty title."""
    cat = meta[meta.price.notna() & (meta.price > 0) & (meta.title.fillna("").str.strip() != "")]
    cat = cat.reset_index(drop=True).copy()
    cat["text"] = [
        catalog_text(t, s, list(f))
        for t, s, f in zip(cat.title, cat.store, cat.features, strict=True)
    ]
    return cat


SLIM_PATH = PROCESSED / "catalog_slim.parquet"
FEATURES_CHARS = 1500


def slim(cat: pd.DataFrame) -> pd.DataFrame:
    """Only what the app needs: card fields, the indexed text, and features as one string."""
    out = cat[
        [
            "parent_asin",
            "title",
            "store",
            "price",
            "average_rating",
            "rating_number",
            "image_url",
            "text",
        ]
    ].copy()
    out["features_text"] = [" ".join(f)[:FEATURES_CHARS] for f in cat.features]
    return out.reset_index(drop=True)


def save_catalog() -> pd.DataFrame:
    cat = build_catalog(pd.read_parquet(PROCESSED / "reviews" / "meta.parquet"))
    cat.to_parquet(CATALOG_PATH)
    slim(cat).to_parquet(SLIM_PATH)
    return cat


if __name__ == "__main__":
    c = save_catalog()
    print(len(c), "catalog items;", f"median price {c.price.median():.2f}")
