import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cartrank.eval.recsys_eval import evaluate, metrics_from_ranks, ranks
from cartrank.recsys.data import build
from cartrank.recsys.models import ItemKNN, Popularity


def _hist(rows, n_items):
    return sp.csr_matrix(
        (np.ones(len(rows)), ([r for r, _ in rows], [c for _, c in rows])),
        shape=(max(r for r, _ in rows) + 1, n_items),
    )


def test_rank_excludes_history_and_is_pessimistic():
    score = np.array([[5.0, 4.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    history = _hist([(0, 0), (1, 3)], 4)  # user 0 already has item 0; user 1 has item 3
    r = ranks(score, np.array([2, 0]), history)
    assert r.tolist() == [1, 2]  # u0: item1 beats target; u1: items 0,1,2 tie, history excluded


def test_metrics_from_ranks():
    m = metrics_from_ranks(np.array([0, 9, 10, 100]))
    assert m["hr@10"] == pytest.approx(0.5)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / np.log2(11)) / 4)


def _toy():
    rows = []
    for u, items in {
        "u1": ["a", "b", "c", "d"],
        "u2": ["a", "b", "c", "e"],
        "u3": ["a", "c", "d", "e"],
    }.items():
        rows += [(u, it, t) for t, it in enumerate(items)]
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])
    from cartrank.data.splits import leave_last_out

    s = leave_last_out(df)
    return build(s["train"], s["dev"], s["test"])


def test_build_and_popularity_eval_runs():
    data = _toy()
    assert data.train.shape == (3, 5)
    assert (data.dev_item >= 0).all() and (data.test_item >= 0).all()
    out = evaluate(Popularity(data.train), data, "dev")
    assert out["all"]["n_users"] == 3
    assert 0.0 <= out["all"]["hr@10"] <= 1.0


def test_dev_history_excludes_train_but_test_history_adds_dev():
    data = _toy()
    assert data.history("dev").nnz == data.train.nnz
    assert data.history("test").nnz == data.train.nnz + 3


def test_itemknn_scores_cooccurring_item_higher():
    train = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float32))
    knn = ItemKNN(train, k=2)
    s = knn.score(np.array([0]))[0]  # user 0 has items 0,1
    assert s[2] > 0
    assert knn.sim.diagonal().sum() == 0
