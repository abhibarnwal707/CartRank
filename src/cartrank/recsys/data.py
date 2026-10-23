"""Recommender data: Office_Products leave-last-out splits as index-mapped sparse matrices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cartrank.data.paths import PROCESSED

COLD_THRESHOLD = 5  # items with fewer train interactions form the cold-start slice


@dataclass
class RecData:
    n_users: int
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray
    train: sp.csr_matrix  # users x items, binary
    dev_item: np.ndarray  # per user, target item index
    test_item: np.ndarray
    train_item_count: np.ndarray  # per item, train interactions

    def history(self, split: str) -> sp.csr_matrix:
        """Items to exclude when evaluating `split`: train for dev, train + dev for test."""
        if split == "dev":
            return self.train
        extra = sp.csr_matrix(
            (np.ones(self.n_users), (np.arange(self.n_users), self.dev_item)),
            shape=self.train.shape,
        )
        return (self.train + extra).tocsr()

    def target(self, split: str) -> np.ndarray:
        return self.dev_item if split == "dev" else self.test_item


def build(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> RecData:
    users = np.sort(pd.concat([train, dev, test]).user_id.unique())
    items = np.sort(pd.concat([train, dev, test]).parent_asin.unique())
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {p: i for i, p in enumerate(items)}

    def idx(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return df.user_id.map(uidx).to_numpy(), df.parent_asin.map(iidx).to_numpy()

    tu, ti = idx(train)
    mat = sp.csr_matrix(
        (np.ones(len(tu), dtype=np.float32), (tu, ti)), shape=(len(users), len(items))
    )
    mat.data[:] = 1.0  # binary even if a (user, item) pair repeats
    targets = []
    for df in (dev, test):
        u, i = idx(df)
        t = np.full(len(users), -1, dtype=np.int64)
        t[u] = i
        targets.append(t)
    return RecData(
        n_users=len(users),
        n_items=len(items),
        user_ids=users,
        item_ids=items,
        train=mat,
        dev_item=targets[0],
        test_item=targets[1],
        train_item_count=np.asarray(mat.sum(axis=0)).ravel(),
    )


def load() -> RecData:
    d = PROCESSED / "reviews"
    return build(*(pd.read_parquet(d / f"{s}.parquet") for s in ("train", "dev", "test")))
