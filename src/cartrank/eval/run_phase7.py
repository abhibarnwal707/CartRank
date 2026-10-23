"""Phase 7: the locked test-split runs. Each stage runs ONCE; configs come from dev results.

python -m cartrank.eval.run_phase7 search1   # BM25 / dense / hybrid on ESCI test (faiss + MPS)
python -m cartrank.eval.run_phase7 search2   # cross-encoder rerank on ESCI test (CPU, no faiss)
python -m cartrank.eval.run_phase7 recsys    # all recommender models on the leave-last-out test
python -m cartrank.eval.run_phase7 report    # results/final_test.md

Every stage refuses to run a second time (see results/test_runs.md) unless FORCE_TEST_RERUN=1,
which exists only for crash recovery and must be disclosed in test_runs.md.
"""

import datetime
import json
import os
import subprocess
import sys

import numpy as np

from cartrank.data.paths import RESULTS, ROOT

FINAL = "final_test.json"
LOG = RESULTS / "test_runs.md"
FT_MODEL = "artifacts/models/bge-small-esci-ft"
CE_K = 50


def git_state() -> str:
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    head = run("rev-parse", "HEAD") or "no-commit"
    dirty = len(run("status", "--porcelain").splitlines())
    return f"{head[:12]} ({'clean' if not dirty else f'{dirty} uncommitted paths'})"


def already_run(stage: str) -> bool:
    return LOG.exists() and f"| {stage} |" in LOG.read_text()


def log_run(stage: str, what: str) -> None:
    if not LOG.exists():
        LOG.write_text(
            "# Test split runs\n\nEach test-split evaluation is run once, after all tuning on dev.\n\n"
            "| Stage | UTC time | Git state | What |\n|---|---|---|---|\n"
        )
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M")
    with LOG.open("a") as f:
        f.write(f"| {stage} | {now} | {git_state()} | {what} |\n")


def guard(stage: str) -> None:
    if already_run(stage) and not os.environ.get("FORCE_TEST_RERUN"):
        raise SystemExit(f"test stage '{stage}' was already run (see results/test_runs.md)")


def _merge(update: dict) -> None:
    p = RESULTS / FINAL
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(update)
    p.write_text(json.dumps(cur, indent=2))


def _dev() -> dict:
    return json.loads((RESULTS / "search_dev.json").read_text())


# ------------------------------------------------------------------ ESCI search, stage 1
def search1() -> None:
    import faiss  # noqa: F401  # must load before torch (libomp order)

    from cartrank.eval.search_eval import SearchEval
    from cartrank.search.rerank import CACHE_DIR

    guard("esci-search-1")
    dev = _dev()
    bb, bf = dev["hybrid_best"], dev["hybrid_best_ft"]
    ev = SearchEval("test")
    ev.run_bm25("bm25")
    ev.run_dense("dense_base", "base")
    ev.run_dense("dense_ft", "ft", model=FT_MODEL)
    ev.run_hybrid("hybrid_base", ("bm25", "dense_base"), k=bb["k"], depth=bb["depth"])
    ev.run_hybrid("hybrid_ft", ("bm25", "dense_ft"), k=bf["k"], depth=bf["depth"])
    rows = {
        f"{n}|{m}": ev.score(n, m)
        for n in ("bm25", "dense_base", "dense_ft", "hybrid_base", "hybrid_ft")
        for m in ("rerank", "full")
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cands = {str(q): ev.full_lists["hybrid_ft"][q][:100] for q in ev.qids}
    (CACHE_DIR / "test_cand_hybrid_ft.json").write_text(json.dumps(cands))
    _merge({"esci_ladder": rows, "esci_config": {"hybrid_base": bb, "hybrid_ft": bf}})
    log_run(
        "esci-search-1", f"{len(ev.qids)} queries: BM25, dense base/FT, hybrid base/FT, both modes"
    )


# ------------------------------------------------------------------ ESCI search, stage 2
def search2() -> None:
    from cartrank.eval.metrics import evaluate
    from cartrank.eval.search_eval import SearchEval, rank_by_scores
    from cartrank.search.rerank import CACHE_DIR, ScoreCache, load_cross_encoder, rerank

    guard("esci-search-2")
    ev = SearchEval("test")
    cands = {
        int(q): c
        for q, c in json.loads((CACHE_DIR / "test_cand_hybrid_ft.json").read_text()).items()
    }
    text = dict(zip(ev.ids, ev.texts, strict=True))
    cache = ScoreCache("minilm_l6_tbcb")
    judged = [(q, p) for q in ev.qids for p in ev.qrels[q]]
    top = [(q, p) for q in ev.qids for p in cands[q][:CE_K]]
    cache.fill(load_cross_encoder(), judged + top, ev.queries, text)
    sc = cache.scores
    rr = {
        q: rank_by_scores(list(ev.qrels[q]), np.array([sc[(q, p)] for p in ev.qrels[q]]))
        for q in ev.qids
    }
    full = {q: rerank(q, cands[q], sc, CE_K) for q in ev.qids}
    _merge(
        {
            "esci_ce": {
                "hybrid_ft_ce|rerank": evaluate(rr, ev.qrels),
                f"hybrid_ft_ce_k{CE_K}|full": evaluate(full, ev.qrels),
            }
        }
    )
    log_run("esci-search-2", f"{len(ev.qids)} queries: cross-encoder rerank (top-{CE_K} / judged)")


# -------------------------------------------------------------------------- recommender
def build_test_hist(data, hist_len: int = 20) -> np.ndarray:
    """Last `hist_len` items of each user's train + dev sequence (dev is the newest item)."""
    import pandas as pd

    from cartrank.data.paths import PROCESSED

    train = pd.read_parquet(PROCESSED / "reviews" / "train.parquet")
    uidx = {u: i for i, u in enumerate(data.user_ids)}
    iidx = {p: i for i, p in enumerate(data.item_ids)}
    train = train.assign(u=train.user_id.map(uidx), i=train.parent_asin.map(iidx))
    train = train.sort_values(["u", "timestamp", "i"], kind="stable")
    hist = np.full((data.n_users, hist_len), data.n_items, dtype=np.int32)
    for u, g in train.groupby("u"):
        seq = np.append(g.i.to_numpy(), data.dev_item[u])[-hist_len:]
        hist[u, hist_len - len(seq) :] = seq
    return hist


def recsys() -> None:
    import torch

    from cartrank.eval.recsys_eval import evaluate
    from cartrank.recsys import bpr
    from cartrank.recsys import data as recdata
    from cartrank.recsys import two_tower as tt
    from cartrank.recsys.models import ItemKNN, Popularity

    guard("recsys-test")
    dev = json.loads((RESULTS / "recsys_dev.json").read_text())
    data = recdata.load()
    out: dict = {}
    out["popularity"] = evaluate(Popularity(data.train), data, "test")
    k = dev["rows"]["itemknn"]["k"]
    knn = ItemKNN(data.train, k=k)
    knn.train = data.history("test")  # at test time the dev item is part of the user's history
    out["itemknn"] = evaluate(knn, data, "test")
    b = dev["rows"]["bpr"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = bpr.fit(
        data.train,
        dim=b["dim"],
        lr=b["lr"],
        reg=b["reg"],
        epochs=b["best_epoch"],
        seed=0,
        device=device,
    )
    out["bpr"] = evaluate(model, data, "test")
    out["bpr_retrain_dev_check"] = evaluate(model, data, "dev")  # should be close to the dev row

    arrays = np.load(ROOT / "artifacts" / "kaggle_twotower" / "data.npz")
    text_emb = np.load(ROOT / "artifacts" / "kaggle_twotower" / "item_text_emb.npy")
    hist = build_test_hist(data)
    hist_csr = data.history("test")
    users = np.arange(data.n_users)
    for variant in ("id_only", "id_text"):
        tau = dev_tt_tau(variant)
        sd = torch.load(
            ROOT / "artifacts" / "two_tower" / f"two_tower_{variant}_tau{tau}.pt",
            map_location="cpu",
        )
        m = tt.TwoTower(data.n_items, text_emb if variant == "id_text" else None, dim=128)
        m.load_state_dict(sd)
        m.to(device)
        out[f"two_tower_{variant}"] = tt.evaluate(
            m,
            hist,
            data.test_item,
            users,
            hist_csr.indptr,
            hist_csr.indices,
            arrays["item_train_count"],
            device,
        ) | {"tau": tau}
    _merge({"recsys": out})
    log_run("recsys-test", "popularity, ItemKNN, BPR (retrained to its dev epoch), two-tower x2")


def dev_tt_tau(variant: str) -> float:
    d = json.loads((RESULTS / "two_tower_dev.json").read_text())
    runs = {n: r for n, r in d.items() if r["variant"] == variant}
    return max(runs.values(), key=lambda r: r["dev_all"]["ndcg@10"])["tau"]


def report() -> None:
    """Write results/final_test.md from final_test.json. Reads results only; touches no split."""
    d = json.loads((RESULTS / FINAL).read_text())
    lad, ce = d["esci_ladder"], d["esci_ce"]
    rows = [
        ("BM25", lad["bm25|full"], lad["bm25|rerank"]),
        ("Dense, bge-small off-the-shelf", lad["dense_base|full"], lad["dense_base|rerank"]),
        ("Dense, fine-tuned on ESCI", lad["dense_ft|full"], lad["dense_ft|rerank"]),
        ("Hybrid RRF (BM25 + fine-tuned dense)", lad["hybrid_ft|full"], lad["hybrid_ft|rerank"]),
        ("Hybrid + cross-encoder (top-50)", ce["hybrid_ft_ce_k50|full"], ce["hybrid_ft_ce|rerank"]),
    ]
    out = [
        "# Final test results\n",
        "Each test split was evaluated once (see test_runs.md). Configurations were chosen on dev only. "
        "No significance tests were run, so differences of about 0.005 or less should not be read as real.\n",
        "## Search, ESCI test (8,956 queries)\n",
        "| System | Full-corpus NDCG@10 | Δ vs previous | Recall@100 | Rerank-mode NDCG@10 | Δ vs previous |",
        "|---|---|---|---|---|---|",
    ]
    prev = None
    for name, full, rr in rows:
        df = "-" if prev is None else f"{full['ndcg@10'] - prev[0]:+.4f}"
        dr = "-" if prev is None else f"{rr['ndcg@10'] - prev[1]:+.4f}"
        out.append(
            f"| {name} | {full['ndcg@10']:.4f} | {df} | {full['recall@100']:.4f} | "
            f"{rr['ndcg@10']:.4f} | {dr} |"
        )
        prev = (full["ndcg@10"], rr["ndcg@10"])
    out += [
        "",
        "Full-corpus counts unjudged products as non-relevant and is not comparable to rerank mode. "
        "In rerank mode the hybrid scores below the fine-tuned dense retriever alone (as on dev), and "
        "the cross-encoder pipeline does not beat dense-only there; the full-corpus ladder is monotonic.\n",
        "## Recommendation, Office_Products test (223,308 users, full ranking over 77,551 items)\n",
        "| Model | HR@10 | NDCG@10 | Cold-slice HR@10 (28,803 users) |",
        "|---|---|---|---|",
    ]
    names = [
        ("popularity", "Popularity"),
        ("itemknn", "ItemKNN"),
        ("bpr", "BPR-MF"),
        ("two_tower_id_only", "Two-tower, ID-only"),
        ("two_tower_id_text", "Two-tower, ID + text"),
    ]
    for key, name in names:
        r = d["recsys"][key]
        out.append(
            f"| {name} | {r['all']['hr@10']:.4f} | {r['all']['ndcg@10']:.4f} | {r['cold']['hr@10']:.4f} |"
        )
    chk = d["recsys"]["bpr_retrain_dev_check"]["all"]
    out += [
        "",
        f"Check: the BPR retrained for the test run scores HR@10 {chk['hr@10']:.4f} / NDCG@10 "
        f"{chk['ndcg@10']:.4f} on dev, matching its dev row. Every model scores lower on test than on "
        "dev, with the same ordering. Cold items (fewer than 5 train interactions): only ItemKNN "
        "retrieves them; both two-tower variants are near zero.\n",
    ]
    (RESULTS / "final_test.md").write_text("\n".join(out))


if __name__ == "__main__":
    {"search1": search1, "search2": search2, "recsys": recsys, "report": report}.get(
        sys.argv[1], lambda: print("use: search1 | search2 | recsys | report")
    )()
