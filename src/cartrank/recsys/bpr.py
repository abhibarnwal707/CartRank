"""BPR matrix factorisation in plain PyTorch (Rendle et al., 2009), uniform negative sampling."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class BPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.user = nn.Parameter(torch.randn(n_users, dim, generator=g) * 0.1)
        self.item = nn.Parameter(torch.randn(n_items, dim, generator=g) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_items))

    def pair_score(self, u, i):
        return (self.user[u] * self.item[i]).sum(-1) + self.bias[i]

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(users, device=self.user.device)
        return (self.user[u] @ self.item.T + self.bias).cpu().numpy().astype(np.float32)


def _sample_negatives(
    rng: np.random.Generator, train: sp.csr_matrix, users: np.ndarray, n_items: int
) -> np.ndarray:
    """Uniform negatives; positives are resampled a few times, any left over are kept (rare)."""
    neg = rng.integers(0, n_items, size=len(users))
    for _ in range(3):
        bad = np.asarray(train[users, neg]).ravel() > 0
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


def fit(
    train: sp.csr_matrix,
    dim: int = 64,
    lr: float = 1e-2,
    reg: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 8192,
    seed: int = 0,
    device: str = "cpu",
    on_epoch=None,
) -> BPR:
    """`on_epoch(epoch, model, mean_loss)` is called after each epoch (e.g. dev evaluation)."""
    coo = train.tocoo()
    users_all, items_all = coo.row.astype(np.int64), coo.col.astype(np.int64)
    model = BPR(train.shape[0], train.shape[1], dim, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(users_all))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            u, i = users_all[idx], items_all[idx]
            j = _sample_negatives(rng, train, u, train.shape[1])
            ut, it, jt = (torch.as_tensor(x, device=device) for x in (u, i, j))
            diff = model.pair_score(ut, it) - model.pair_score(ut, jt)
            loss = -nn.functional.logsigmoid(diff).mean()
            # L2 on the embeddings actually used in this batch
            penalty = (
                reg
                * (
                    model.user[ut].pow(2).sum()
                    + model.item[it].pow(2).sum()
                    + model.item[jt].pow(2).sum()
                )
                / len(idx)
            )
            opt.zero_grad()
            (loss + penalty).backward()
            opt.step()
            losses.append(loss.item())
        if on_epoch is not None:
            on_epoch(epoch, model, float(np.mean(losses)))
    return model
