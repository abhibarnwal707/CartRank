import numpy as np
import pandas as pd


def split_queries(
    query_ids: pd.Series, dev_frac: float = 0.1, seed: int = 0
) -> tuple[set[int], set[int]]:
    """Split by query (never by row). Returns (train_ids, dev_ids)."""
    ids = np.sort(query_ids.unique())
    rng = np.random.default_rng(seed)
    n_dev = int(round(len(ids) * dev_frac))
    dev = set(rng.choice(ids, size=n_dev, replace=False).tolist())
    train = set(ids.tolist()) - dev
    return train, dev


def leave_last_out(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
    min_interactions: int = 3,
) -> dict[str, pd.DataFrame]:
    """Temporal leave-last-out per user: last = test, second-to-last = dev, rest = train.

    Users with fewer than `min_interactions` interactions are dropped. Ties on timestamp
    are broken by item id so the split is deterministic.
    """
    df = df.sort_values([user_col, time_col, item_col], kind="stable")
    counts = df.groupby(user_col)[item_col].transform("size")
    df = df[counts >= min_interactions].copy()
    # rank from the end: 0 = last, 1 = second to last
    rank_from_end = df.groupby(user_col).cumcount(ascending=False)
    return {
        "train": df[rank_from_end >= 2],
        "dev": df[rank_from_end == 1],
        "test": df[rank_from_end == 0],
    }
