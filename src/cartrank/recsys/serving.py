"""Numpy-only serving of the trained two-tower model (no torch needed in the app).

Export once from a training checkpoint; at runtime load item vectors + the small user MLP.
user vector = normalize(MLP(mean of raw item vectors of the liked items)), exactly as in
two_tower.TwoTower.user_vec.
"""

import numpy as np

from cartrank.data.paths import ROOT

SERVING_DIR = ROOT / "artifacts" / "serving"


def export_from_state_dict(sd: dict, n_items: int) -> dict[str, np.ndarray]:
    """Precompute item_raw = id_emb + text_proj(text) (drops the padding row) and copy the MLP."""
    raw = sd["id_emb.weight"][:n_items].float()
    if "text_proj.weight" in sd:
        raw = raw + sd["text"][:n_items].float() @ sd["text_proj.weight"].float().T
    out = {"item_raw": raw.numpy().astype(np.float32)}
    for k in ("user_mlp.0.weight", "user_mlp.0.bias", "user_mlp.2.weight", "user_mlp.2.bias"):
        out[k] = sd[k].float().numpy().astype(np.float32)
    return out


def export(checkpoint: str, out_dir=SERVING_DIR) -> None:
    import torch

    from cartrank.recsys import data as recdata

    data = recdata.load()
    sd = torch.load(checkpoint, map_location="cpu")
    arrays = export_from_state_dict(sd, data.n_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "two_tower_serving.npz", **arrays)
    (out_dir / "item_ids.txt").write_text("\n".join(data.item_ids))
    print({k: v.shape for k, v in arrays.items()})


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class TwoTowerServing:
    def __init__(self, arrays: dict[str, np.ndarray], item_ids: list[str]):
        self.raw = arrays["item_raw"]
        self.vecs = _normalize(self.raw)
        self.w0, self.b0 = arrays["user_mlp.0.weight"], arrays["user_mlp.0.bias"]
        self.w2, self.b2 = arrays["user_mlp.2.weight"], arrays["user_mlp.2.bias"]
        self.item_ids = item_ids
        self.index = {a: i for i, a in enumerate(item_ids)}

    @classmethod
    def load(cls, directory=SERVING_DIR) -> "TwoTowerServing":
        arrays = dict(np.load(directory / "two_tower_serving.npz"))
        return cls(arrays, (directory / "item_ids.txt").read_text().split("\n"))

    def user_vec(self, liked: list[str]) -> np.ndarray:
        idx = [self.index[a] for a in liked if a in self.index]
        if not idx:
            raise ValueError("none of the liked items is in the recommender vocabulary")
        mean = self.raw[idx].mean(axis=0)
        h = np.maximum(mean @ self.w0.T + self.b0, 0.0)
        return _normalize(h @ self.w2.T + self.b2)

    def _top(self, scores: np.ndarray, k: int, allowed: set[str] | None, skip: set[str]):
        order = np.argsort(-scores)
        out = []
        for i in order:
            a = self.item_ids[i]
            if a in skip or (allowed is not None and a not in allowed):
                continue
            out.append((a, float(scores[i])))
            if len(out) == k:
                break
        return out

    def recommend(self, liked: list[str], k: int = 10, allowed: set[str] | None = None):
        """Top-k items for a user who liked `liked` (excluded from the result)."""
        return self._top(self.vecs @ self.user_vec(liked), k, allowed, set(liked))

    def similar(self, asin: str, k: int = 10, allowed: set[str] | None = None):
        """Top-k items nearest to `asin` in the item-vector space (cosine), excluding itself."""
        if asin not in self.index:
            raise ValueError("item is not in the recommender vocabulary")
        return self._top(self.vecs @ self.vecs[self.index[asin]], k, allowed, {asin})
