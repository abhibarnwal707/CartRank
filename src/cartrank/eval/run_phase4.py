"""Phase 4 driver: python -m cartrank.eval.run_phase4 <baselines>   (ESCI-independent, dev only)."""

import itertools
import json
import sys
import time

import numpy as np
import torch

from cartrank.data.paths import RESULTS
from cartrank.eval.recsys_eval import evaluate
from cartrank.eval.search_eval import merge_results
from cartrank.recsys import bpr
from cartrank.recsys import data as recdata
from cartrank.recsys.models import ItemKNN, Popularity

JSON = "recsys_dev.json"


def _rows() -> dict:
    p = RESULTS / JSON
    return json.loads(p.read_text()).get("rows", {}) if p.exists() else {}


def baselines() -> None:
    data = recdata.load()
    print(f"{data.n_users} users, {data.n_items} items, {data.train.nnz} train interactions")
    rows = _rows()

    t0 = time.time()
    pop = Popularity(data.train)
    rows["popularity"] = {**evaluate(pop, data, "dev"), "train_seconds": time.time() - t0}
    print("popularity", rows["popularity"]["all"])

    knn_tuning = []
    for k in (50, 100, 200):
        t0 = time.time()
        knn = ItemKNN(data.train, k=k)
        fit = time.time() - t0
        res = evaluate(knn, data, "dev")
        knn_tuning.append({"k": k, "train_seconds": fit, **res})
        print("itemknn", k, res["all"])
    best = max(knn_tuning, key=lambda r: r["all"]["ndcg@10"])
    rows["itemknn"] = {**best}
    merge_results(
        JSON,
        {
            "rows": rows,
            "itemknn_tuning": knn_tuning,
            "data": {
                "users": data.n_users,
                "items": data.n_items,
                "train_interactions": int(data.train.nnz),
            },
        },
    )


def run_bpr(long: bool = False) -> None:
    """Tune BPR on dev. Per-epoch selection uses a fixed 20,000-user dev subset (every 2 epochs,
    patience 3 evals); the chosen model is then scored on ALL dev users."""
    data = recdata.load()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(0)
    subset = np.sort(rng.choice(np.flatnonzero(data.dev_item >= 0), 20_000, replace=False))
    if long:
        # the first sweep's best settings hit the 60-epoch cap while still improving
        grid, max_epochs = [(128, 3e-3, 1e-3), (128, 5e-3, 3e-3)], 250
    else:
        grid, max_epochs = list(itertools.product((64, 128), (3e-3, 1e-2), (1e-4, 1e-3, 1e-2))), 60
    results, best = [], None
    for dim, lr, reg in grid:
        state = {"best": -1.0, "epoch": 0, "bad": 0, "weights": None}

        def on_epoch(epoch, model, loss, state=state):
            if epoch % 2:
                return None
            ndcg = evaluate(model, data, "dev", users=subset)["all"]["ndcg@10"]
            if ndcg > state["best"]:
                state.update(
                    best=ndcg,
                    epoch=epoch,
                    bad=0,
                    weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            else:
                state["bad"] += 1
            return state["bad"] >= 3 or None

        t0 = time.time()
        model = bpr.fit(
            data.train, dim=dim, lr=lr, reg=reg, epochs=max_epochs, device=device, on_epoch=on_epoch
        )
        secs = time.time() - t0
        row = {
            "dim": dim,
            "lr": lr,
            "reg": reg,
            "best_epoch": state["epoch"],
            "subset_ndcg@10": state["best"],
            "train_seconds": secs,
        }
        results.append(row)
        print(row, flush=True)
        if best is None or state["best"] > best["subset_ndcg@10"]:
            best = {**row, "weights": state["weights"]}
            best_model = model
            best_model.load_state_dict(state["weights"])
    full = evaluate(best_model, data, "dev")
    cfg = {k: v for k, v in best.items() if k != "weights"}
    print("best", cfg, full["all"], flush=True)
    rows = _rows()
    prev = rows.get("bpr")
    update = {"bpr_tuning_long" if long else "bpr_tuning": results}
    if prev is None or cfg["subset_ndcg@10"] > prev["subset_ndcg@10"]:
        rows["bpr"] = {**full, **cfg}
    merge_results(JSON, {"rows": rows, **update})


if __name__ == "__main__":
    {
        "baselines": baselines,
        "bpr": run_bpr,
        "bpr-long": lambda: run_bpr(long=True),
    }[sys.argv[1]]()
