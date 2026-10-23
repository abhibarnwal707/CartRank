import numpy as np
import scipy.sparse as sp

from cartrank.recsys import bpr


def _block_data():
    # two user groups with disjoint tastes: users 0-9 like items 0-4, users 10-19 like items 5-9
    rows, cols = [], []
    rng = np.random.default_rng(0)
    for u in range(20):
        base = 0 if u < 10 else 5
        for i in rng.choice(5, size=3, replace=False):
            rows.append(u)
            cols.append(base + i)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(20, 10))


def test_bpr_learns_block_structure_and_loss_falls():
    train = _block_data()
    losses = []
    model = bpr.fit(
        train,
        dim=8,
        lr=0.05,
        reg=0.0,
        epochs=40,
        batch_size=16,
        seed=0,
        on_epoch=lambda e, m, loss: losses.append(loss),
    )
    assert losses[-1] < losses[0]
    s = model.score(np.arange(20))
    # users of group A prefer group-A items on average, and likewise for group B
    assert s[:10, :5].mean() > s[:10, 5:].mean()
    assert s[10:, 5:].mean() > s[10:, :5].mean()


def test_negative_sampler_avoids_positives():
    train = _block_data()
    rng = np.random.default_rng(1)
    users = np.zeros(2000, dtype=np.int64)
    neg = bpr._sample_negatives(rng, train, users, 10)
    positives = set(train[0].indices.tolist())
    assert sum(n in positives for n in neg) < 20  # a few misses are tolerated by design
