"""Full-ranking recommender evaluation: HR@10 and NDCG@10 over ALL items, no sampled negatives.

Items the user already interacted with (train, plus dev when scoring test) are excluded from
the candidates. Ties with the target are counted against the model (pessimistic rank), so a model
that scores many items identically (e.g. popularity on cold items) cannot gain from tie order.
"""

import numpy as np

from cartrank.recsys.data import COLD_THRESHOLD, RecData

K = 10


def ranks(score: np.ndarray, target: np.ndarray, history) -> np.ndarray:
    """0-based pessimistic rank of each row's target among non-history items."""
    s = score.copy()
    rows = np.repeat(np.arange(s.shape[0]), np.diff(history.indptr))
    s[rows, history.indices] = -np.inf
    t = s[np.arange(s.shape[0]), target]
    return (s >= t[:, None]).sum(axis=1) - 1  # minus the target itself


def metrics_from_ranks(r: np.ndarray, k: int = K) -> dict[str, float]:
    hit = r < k
    return {
        "hr@10": float(hit.mean()),
        "ndcg@10": float((hit / np.log2(r + 2)).mean()),
        "n_users": int(len(r)),
    }


def evaluate(model, data: RecData, split: str = "dev", batch: int = 512) -> dict:
    target = data.target(split)
    history = data.history(split)
    users = np.flatnonzero(target >= 0)
    all_r = np.empty(len(users), dtype=np.int64)
    for start in range(0, len(users), batch):
        u = users[start : start + batch]
        # history rows must align with the batch: slice then rank
        all_r[start : start + len(u)] = ranks(model.score(u), target[u], history[u])
    cold = data.train_item_count[target[users]] < COLD_THRESHOLD
    return {
        "all": metrics_from_ranks(all_r),
        "cold": metrics_from_ranks(all_r[cold]) if cold.any() else None,
    }
