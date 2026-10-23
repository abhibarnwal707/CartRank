import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cartrank.eval.recsys_eval import ranks as np_ranks
from cartrank.recsys import two_tower as tt


def _toy(n_users=40, n_items=12, L=4, seed=0):
    rng = np.random.default_rng(seed)
    csr = sp.csr_matrix((rng.random((n_users, n_items)) < 0.25).astype(np.float32))
    target = rng.integers(0, n_items, n_users)
    hist = rng.integers(0, n_items + 1, (n_users, L))  # n_items == padding
    return csr, target, hist


def test_evaluate_matches_numpy_full_ranking():
    csr, target, hist = _toy()
    model = tt.TwoTower(12, None, dim=8, hidden=16)
    out = tt.evaluate(
        model,
        hist,
        target,
        np.arange(40),
        csr.indptr,
        csr.indices,
        np.asarray(csr.sum(0)).ravel(),
        "cpu",
        batch=7,
    )
    with torch.no_grad():
        scores = (model.user_vec(torch.as_tensor(hist)) @ model.all_item_vecs().T).numpy()
    r = np_ranks(scores, target, csr)
    assert out["all"]["hr@10"] == pytest.approx((r < 10).mean())
    assert out["all"]["ndcg@10"] == pytest.approx(((r < 10) / np.log2(r + 2)).mean())


def test_loss_masks_duplicate_targets():
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    v = u.clone()
    target = torch.tensor([3, 3, 5, 6])
    log_q = torch.zeros(10)
    loss = tt.sampled_softmax_loss(u, v, target, log_q, 0.1)
    assert torch.isfinite(loss)  # row 0's logit for the duplicate (row 1) is -inf, not a negative


def test_padding_is_ignored_in_user_vector():
    model = tt.TwoTower(5, np.random.default_rng(0).random((5, 6)).astype(np.float32), dim=8)
    a = model.user_vec(torch.tensor([[1, 2, 5, 5]]))
    b = model.user_vec(torch.tensor([[1, 2, 5, 5, 5, 5]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_train_improves_on_block_data():
    n_items, n_users, L = 10, 60, 4
    rng = np.random.default_rng(0)
    hist, target, user_rows = [], [], []
    for u in range(n_users):
        base = 0 if u < 30 else 5
        seq = base + rng.integers(0, 5, 6)
        for t in range(1, 6):
            h = list(seq[max(0, t - L) : t])[-L:]
            hist.append([n_items] * (L - len(h)) + h)
            target.append(seq[t])
            user_rows.append(u)
    hist, target = np.array(hist), np.array(target)
    tr = sp.csr_matrix((np.ones(len(target)), (user_rows, target)), shape=(n_users, n_items))
    tr.data[:] = 1.0
    dev_t = np.array([0 if u < 30 else 5 for u in range(n_users)])
    dev_h = np.array([[n_items, n_items, n_items, 1 if u < 30 else 6] for u in range(n_users)])
    data = {
        "n_items": n_items,
        "train_hist": hist,
        "train_target": target,
        "item_train_count": np.asarray(tr.sum(0)).ravel(),
        "train_indptr": tr.indptr,
        "train_indices": tr.indices,
        "dev_hist": dev_h,
        "dev_target": dev_t,
    }
    model, rec = tt.train(
        data, None, "cpu", dim=8, batch=16, epochs=15, patience=15, lr=1e-2, log=lambda *_: None
    )
    assert rec[-1]["loss"] < rec[0]["loss"]
    assert max(r["all"]["hr@10"] for r in rec) > 0.5
