"""Two-tower recommender: torch + numpy only, so this file can be pasted into a Kaggle notebook.

Item tower: ID embedding, optionally plus a projection of the item's text embedding (cold items).
User tower: masked mean of the (shared) item vectors of the user's recent items, then an MLP.
Loss: in-batch sampled softmax with log-q popularity correction. Evaluation: full ranking over
all items, history excluded, pessimistic ties (same rule as cartrank.eval.recsys_eval).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

COLD_THRESHOLD = 5


class TwoTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        text_emb: np.ndarray | None = None,
        dim: int = 128,
        hidden: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_items = n_items
        self.id_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # last row = padding
        nn.init.normal_(self.id_emb.weight, std=0.1)
        with torch.no_grad():
            self.id_emb.weight[n_items].zero_()
        self.has_text = text_emb is not None
        if self.has_text:
            text = torch.cat(
                [torch.as_tensor(text_emb, dtype=torch.float32), torch.zeros(1, text_emb.shape[1])]
            )
            self.register_buffer("text", text)
            self.text_proj = nn.Linear(text_emb.shape[1], dim, bias=False)
        self.user_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def item_raw(self, idx: torch.Tensor) -> torch.Tensor:
        v = self.id_emb(idx)
        if self.has_text:
            v = v + self.text_proj(self.text[idx])
        return v

    def item_vec(self, idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_raw(idx), dim=-1)

    def user_vec(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: (B, L) item indices, padded with n_items."""
        mask = (hist != self.n_items).float().unsqueeze(-1)
        mean = (self.item_raw(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(self.user_mlp(mean), dim=-1)

    @torch.no_grad()
    def all_item_vecs(self, chunk: int = 16384) -> torch.Tensor:
        dev = self.id_emb.weight.device
        out = [
            self.item_vec(torch.arange(s, min(s + chunk, self.n_items), device=dev))
            for s in range(0, self.n_items, chunk)
        ]
        return torch.cat(out)


def sampled_softmax_loss(u, v, target, log_q, tau):
    """In-batch softmax; logits[i, j] = u_i . v_j / tau - log q(target_j).

    Other rows holding the same target item are masked out (they are not negatives).
    """
    logits = (u @ v.T) / tau - log_q[target][None, :]
    same = target[:, None] == target[None, :]
    eye = torch.eye(len(target), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(target), device=u.device))


def _history_mask(indptr: np.ndarray, indices: np.ndarray, users: np.ndarray):
    """Row/column positions of the history items of `users` (CSR gather, vectorised)."""
    lens = indptr[users + 1] - indptr[users]
    rows = np.repeat(np.arange(len(users)), lens)
    offsets = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    cols = indices[np.repeat(indptr[users], lens) + offsets]
    return rows, cols


@torch.no_grad()
def evaluate(
    model,
    hist,
    target,
    user_ids,
    indptr,
    indices,
    item_count,
    device,
    batch: int = 2048,
    k: int = 10,
) -> dict:
    """Full-ranking HR@k / NDCG@k over all items for every row of (hist, target).

    `user_ids` index the CSR history to exclude (train for dev). Cold slice = target items with
    fewer than COLD_THRESHOLD train interactions.
    """
    model.eval()
    V = model.all_item_vecs()
    ranks = []
    for s in range(0, len(target), batch):
        h = torch.as_tensor(hist[s : s + batch], dtype=torch.long, device=device)
        scores = model.user_vec(h) @ V.T
        rows, cols = _history_mask(indptr, indices, user_ids[s : s + batch])
        scores[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = float(
            "-inf"
        )
        t = torch.as_tensor(target[s : s + batch], dtype=torch.long, device=device)
        ts = scores.gather(1, t[:, None])
        ranks.append(((scores >= ts).sum(1) - 1).cpu().numpy())
    r = np.concatenate(ranks)

    def metrics(x: np.ndarray) -> dict:
        hit = x < k
        return {
            "hr@10": float(hit.mean()),
            "ndcg@10": float((hit / np.log2(x + 2)).mean()),
            "n_users": int(len(x)),
        }

    cold = item_count[target] < COLD_THRESHOLD
    return {"all": metrics(r), "cold": metrics(r[cold]) if cold.any() else None}


def train(
    data: dict,
    text_emb: np.ndarray | None,
    device: str,
    dim: int = 128,
    tau: float = 0.05,
    lr: float = 1e-3,
    batch: int = 2048,
    epochs: int = 20,
    patience: int = 3,
    seed: int = 0,
    log=print,
) -> tuple[nn.Module, list[dict]]:
    """Train on `data` (see package_kaggle.build_arrays), selecting the epoch by dev NDCG@10.

    Returns the best-epoch model and per-epoch records. Only train and dev data are used.
    """
    n_items = int(data["n_items"])
    model = TwoTower(n_items, text_emb, dim=dim, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    count = data["item_train_count"].astype(np.float64)
    log_q = torch.as_tensor(
        np.log((count + 1) / (count.sum() + n_items)), dtype=torch.float32, device=device
    )
    X, y = data["train_hist"], data["train_target"]
    rng = np.random.default_rng(seed)
    dev_users = np.arange(len(data["dev_target"]))
    best, best_state, bad, records = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(y))
        losses = []
        for s in range(0, len(order) - batch + 1, batch):  # drop the ragged last batch
            idx = order[s : s + batch]
            h = torch.as_tensor(X[idx], dtype=torch.long, device=device)
            t = torch.as_tensor(y[idx], dtype=torch.long, device=device)
            loss = sampled_softmax_loss(model.user_vec(h), model.item_vec(t), t, log_q, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_s = time.time() - t0
        res = evaluate(
            model,
            data["dev_hist"],
            data["dev_target"],
            dev_users,
            data["train_indptr"],
            data["train_indices"],
            data["item_train_count"],
            device,
        )
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "train_seconds": train_s, **res}
        records.append(rec)
        log(
            f"epoch {epoch}: loss {rec['loss']:.4f} dev NDCG@10 {res['all']['ndcg@10']:.4f} "
            f"HR@10 {res['all']['hr@10']:.4f} ({train_s:.0f}s)"
        )
        if res["all"]["ndcg@10"] > best:
            best, bad = res["all"]["ndcg@10"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, records
