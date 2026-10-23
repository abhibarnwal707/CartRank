import pandas as pd

from cartrank.data.paths import ESCI_DIR, PROCESSED
from cartrank.data.splits import split_queries

EXAMPLES = ESCI_DIR / "shopping_queries_dataset_examples.parquet"
PRODUCTS = ESCI_DIR / "shopping_queries_dataset_products.parquet"


def load_examples() -> pd.DataFrame:
    """ESCI Task 1 ranking set: US locale, small_version == 1, provided train/test split."""
    ex = pd.read_parquet(EXAMPLES)
    return ex[(ex.product_locale == "us") & (ex.small_version == 1)].reset_index(drop=True)


def load_products(product_ids: pd.Series | None = None) -> pd.DataFrame:
    cols = [
        "product_id",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    prod = pd.read_parquet(PRODUCTS, columns=["product_locale", *cols])
    prod = prod[prod.product_locale == "us"].drop(columns="product_locale")
    if product_ids is not None:
        prod = prod[prod.product_id.isin(set(product_ids))]
    return prod.reset_index(drop=True)


def make_splits(seed: int = 0) -> dict[str, pd.DataFrame]:
    """train / dev (10% of train query_ids) / test (provided, untouched until Phase 7)."""
    ex = load_examples()
    train_all = ex[ex.split == "train"]
    train_ids, dev_ids = split_queries(train_all.query_id, dev_frac=0.1, seed=seed)
    return {
        "train": train_all[train_all.query_id.isin(train_ids)],
        "dev": train_all[train_all.query_id.isin(dev_ids)],
        "test": ex[ex.split == "test"],
    }


def save_splits(seed: int = 0) -> None:
    out = PROCESSED / "esci"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in make_splits(seed).items():
        df.reset_index(drop=True).to_parquet(out / f"{name}.parquet")
