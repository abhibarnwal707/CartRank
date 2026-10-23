"""Build the Kaggle package for the two-tower notebook.

python -m cartrank.recsys.package_kaggle arrays     -> artifacts/kaggle_twotower/data.npz
python -m cartrank.recsys.package_kaggle text <encoder_path>  -> item_text_emb.npy
python -m cartrank.recsys.package_kaggle notebook   -> notebooks/kaggle/twotower-office.ipynb

Only TRAIN and DEV data are packaged. The test split is deliberately left out: it is run once,
in Phase 7.
"""

import json
import sys

import numpy as np
import pandas as pd

from cartrank.data.catalog import catalog_text
from cartrank.data.paths import PROCESSED, ROOT
from cartrank.recsys import data as recdata

OUT = ROOT / "artifacts" / "kaggle_twotower"
NOTEBOOK = ROOT / "notebooks" / "kaggle" / "twotower-office.ipynb"
HIST_LEN = 20


def build_arrays(hist_len: int = HIST_LEN) -> dict:
    data = recdata.load()
    train = pd.read_parquet(PROCESSED / "reviews" / "train.parquet")
    uidx = {u: i for i, u in enumerate(data.user_ids)}
    iidx = {p: i for i, p in enumerate(data.item_ids)}
    train = train.assign(u=train.user_id.map(uidx), i=train.parent_asin.map(iidx))
    train = train.sort_values(["u", "timestamp", "i"], kind="stable")
    pad = data.n_items
    seqs = {u: g.i.to_numpy() for u, g in train.groupby("u")}

    n_examples = sum(max(len(s) - 1, 0) for s in seqs.values())
    x = np.full((n_examples, hist_len), pad, dtype=np.int32)
    y = np.empty(n_examples, dtype=np.int32)
    dev_hist = np.full((data.n_users, hist_len), pad, dtype=np.int32)
    row = 0
    for u, s in seqs.items():
        for t in range(1, len(s)):
            h = s[max(0, t - hist_len) : t]
            x[row, hist_len - len(h) :] = h  # left-pad so the newest item is last
            y[row] = s[t]
            row += 1
        h = s[-hist_len:]
        dev_hist[u, hist_len - len(h) :] = h
    assert row == n_examples
    return {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "train_hist": x,
        "train_target": y,
        "dev_hist": dev_hist,
        "dev_target": data.dev_item.astype(np.int32),
        "train_indptr": data.train.indptr.astype(np.int64),
        "train_indices": data.train.indices.astype(np.int32),
        "item_train_count": data.train_item_count.astype(np.int32),
    }


def save_arrays() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = build_arrays()
    assert (arrays["dev_target"] >= 0).all()
    np.savez_compressed(OUT / "data.npz", **arrays)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in arrays.items()})


def save_text_embeddings(encoder_path: str) -> None:
    from cartrank.search.dense import load_encoder

    data = recdata.load()
    meta = pd.read_parquet(PROCESSED / "reviews" / "meta.parquet").set_index("parent_asin")
    texts = [
        catalog_text(m.title, m.store, list(m.features))
        for m in (meta.loc[a] for a in data.item_ids)
    ]
    enc = load_encoder(encoder_path)
    vecs = enc.encode(
        texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "item_text_emb.npy", vecs)
    print("saved", vecs.shape)


def _cell(kind: str, src: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def make_notebook() -> None:
    module_src = (ROOT / "src" / "cartrank" / "recsys" / "two_tower.py").read_text()
    cells = [
        _cell(
            "markdown",
            """
# CartRank: two-tower recommender (Kaggle)

Trains the two-tower model on Amazon Reviews 2023 Office_Products (temporal leave-last-out
splits, built in the CartRank repo) and runs the ablation **ID-only vs ID + text embedding**
for the item tower, scored with full-ranking HR@10 / NDCG@10 over all items on the **dev** split,
with a cold-start slice (items with fewer than 5 train interactions).

Use a **GPU** session. Internet is not needed. Attach the Kaggle Dataset that holds
`data.npz` and `item_text_emb.npy`. Only train and dev data are included; the test split is
deliberately absent.
""",
        ),
        _cell(
            "code",
            """
import glob, json, os, time
import numpy as np, torch

npz = glob.glob("/kaggle/input/**/data.npz", recursive=True)[0]
txt = glob.glob("/kaggle/input/**/item_text_emb.npy", recursive=True)[0]
data = dict(np.load(npz))
text_emb = np.load(txt)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, torch.__version__, {k: v.shape for k, v in data.items() if hasattr(v, "shape") and v.ndim})
print("users", int(data["n_users"]), "items", int(data["n_items"]), "text", text_emb.shape)
assert text_emb.shape[0] == int(data["n_items"])
""",
        ),
        _cell(
            "markdown",
            "## Model code (identical to `src/cartrank/recsys/two_tower.py` in the repo)",
        ),
        _cell("code", "%%writefile two_tower.py\n" + module_src),
        _cell("code", "import two_tower as tt"),
        _cell(
            "markdown",
            """
## Runs
Four runs: item tower ID-only vs ID + text, each at temperature 0.05 and 0.1. The epoch is chosen
by dev NDCG@10 (patience 3). Edit `EPOCHS`/`DIM` here if you want a different budget.
""",
        ),
        _cell(
            "code",
            """
EPOCHS, DIM, BATCH, LR = 20, 128, 2048, 1e-3
results, models = {}, {}
for variant, emb in (("id_only", None), ("id_text", text_emb)):
    for tau in (0.05, 0.1):
        name = f"{variant}_tau{tau}"
        print("=== ", name)
        t0 = time.time()
        model, records = tt.train(data, emb, device, dim=DIM, tau=tau, lr=LR, batch=BATCH,
                                  epochs=EPOCHS)
        best = max(records, key=lambda r: r["all"]["ndcg@10"])
        results[name] = {"variant": variant, "tau": tau, "dim": DIM, "best_epoch": best["epoch"],
                         "dev_all": best["all"], "dev_cold": best["cold"],
                         "train_seconds_total": time.time() - t0, "epochs": records}
        torch.save(model.state_dict(), f"/kaggle/working/two_tower_{name}.pt")
        json.dump(results, open("/kaggle/working/two_tower_results.json", "w"), indent=1)
""",
        ),
        _cell(
            "code",
            """
print(f"{'run':24s} {'epoch':>5s} {'HR@10':>8s} {'NDCG@10':>8s} {'cold HR@10':>11s} {'cold NDCG@10':>13s}")
for n, r in results.items():
    c = r["dev_cold"] or {"hr@10": float("nan"), "ndcg@10": float("nan")}
    print(f"{n:24s} {r['best_epoch']:5d} {r['dev_all']['hr@10']:8.4f} {r['dev_all']['ndcg@10']:8.4f} "
          f"{c['hr@10']:11.4f} {c['ndcg@10']:13.4f}")
""",
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print("wrote", NOTEBOOK)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "arrays":
        save_arrays()
    elif cmd == "text":
        save_text_embeddings(sys.argv[2])
    else:
        make_notebook()
