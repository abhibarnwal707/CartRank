import json

import pandas as pd

from cartrank.data.paths import CATEGORY, PROCESSED, REVIEWS_DIR
from cartrank.data.splits import leave_last_out

LAST_OUT = REVIEWS_DIR / "last_out"
META_JSONL = REVIEWS_DIR / "meta" / f"meta_{CATEGORY}.jsonl"


def load_official(category: str = CATEGORY) -> dict[str, pd.DataFrame]:
    """Official 5-core last_out benchmark files (train/valid/test)."""
    return {s: pd.read_csv(LAST_OUT / f"{category}.{s}.csv") for s in ("train", "valid", "test")}


def load_interactions(category: str = CATEGORY) -> pd.DataFrame:
    return pd.concat(load_official(category).values(), ignore_index=True)


def _first_image(images: list[dict] | None) -> str | None:
    """`images` is a list of {thumb, large, hi_res, variant} dicts; prefer the MAIN variant."""
    if not images:
        return None
    ordered = sorted(images, key=lambda im: im.get("variant") != "MAIN")
    for im in ordered:
        for key in ("large", "hi_res", "thumb"):
            if im.get(key):
                return im[key]
    return None


def _price(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def extract_meta(item_ids: set[str]) -> pd.DataFrame:
    """Stream the raw meta jsonl and keep only the catalog items we need."""
    rows = []
    with open(META_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            asin = rec.get("parent_asin")
            if asin not in item_ids:
                continue
            rows.append(
                {
                    "parent_asin": asin,
                    "title": rec.get("title"),
                    "store": rec.get("store"),
                    "price": _price(rec.get("price")),
                    "average_rating": rec.get("average_rating"),
                    "rating_number": rec.get("rating_number"),
                    "categories": rec.get("categories") or [],
                    "features": rec.get("features") or [],
                    "image_url": _first_image(rec.get("images")),
                }
            )
    return pd.DataFrame(rows)


def make_splits(min_interactions: int = 3) -> dict[str, pd.DataFrame]:
    return leave_last_out(load_interactions(), min_interactions=min_interactions)


def save_all() -> None:
    out = PROCESSED / "reviews"
    out.mkdir(parents=True, exist_ok=True)
    splits = make_splits()
    for name, df in splits.items():
        df.reset_index(drop=True).to_parquet(out / f"{name}.parquet")
    items = set(pd.concat(splits.values()).parent_asin)
    extract_meta(items).to_parquet(out / "meta.parquet")
