"""Phase 4 driver: python -m cartrank.eval.run_phase4 <baselines>   (ESCI-independent, dev only)."""

import json
import sys
import time

from cartrank.data.paths import RESULTS
from cartrank.eval.recsys_eval import evaluate
from cartrank.eval.search_eval import merge_results
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


if __name__ == "__main__":
    {"baselines": baselines}[sys.argv[1]]()
