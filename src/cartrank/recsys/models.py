"""Recommender baselines. Every model exposes score(users) -> dense (len(users), n_items)."""

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


class Popularity:
    """Train-period interaction counts; the same ranking for every user."""

    def __init__(self, train: sp.csr_matrix):
        self.counts = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.counts, (len(users), 1))


class ItemKNN:
    """Item-item cosine similarity on the binary interaction matrix, top-k neighbours per item.

    score(u, j) = sum over the user's train items i of sim(i, j).
    """

    def __init__(self, train: sp.csr_matrix, k: int = 100, shrink: float = 0.0):
        self.train = train
        item_user = normalize(train.T.tocsr().astype(np.float32), axis=1)  # unit item vectors
        n = item_user.shape[0]
        rows, cols, vals = [], [], []
        block = 2000
        for start in range(0, n, block):
            sim = (item_user[start : start + block] @ item_user.T).toarray()
            if shrink:
                sim = sim / (1.0 + shrink)
            sim[np.arange(sim.shape[0]), np.arange(start, start + sim.shape[0])] = 0.0  # no self
            kk = min(k, n - 1)
            top = np.argpartition(-sim, kk, axis=1)[:, :kk]
            v = np.take_along_axis(sim, top, axis=1)
            keep = v > 0
            r = np.repeat(np.arange(start, start + sim.shape[0]), kk).reshape(-1, kk)
            rows.append(r[keep])
            cols.append(top[keep])
            vals.append(v[keep])
        self.sim = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.train[users] @ self.sim).toarray().astype(np.float32)
