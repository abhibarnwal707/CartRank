"""Phase 1 driver: python -m cartrank.eval.run_phase1 <bm25-ablation|dense-ablation|main|report>

Everything is evaluated on ESCI dev. Test is not touched.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from cartrank.data.paths import RESULTS
from cartrank.eval.search_eval import SearchEval, latency_ms, merge_results
from cartrank.search.corpus import VARIANTS
from cartrank.search.dense import (
    BASE_MODEL,
    encode_queries,
    load_encoder,
)

JSON = "search_dev.json"
LADDER = [
    ("bm25", "BM25"),
    ("dense_base", "Dense (bge-small, off-the-shelf)"),
    ("hybrid_base", "Hybrid RRF (BM25 + dense)"),
]


def _load() -> dict:
    p = RESULTS / JSON
    return json.loads(p.read_text()) if p.exists() else {}


def bm25_ablation() -> None:
    out = {}
    for v in VARIANTS:
        ev = SearchEval("dev", variant=v)
        ev.run_bm25()
        out[v] = {m: ev.score("bm25", m) for m in ("rerank", "full")}
        print(v, {m: round(out[v][m]["ndcg@10"], 4) for m in out[v]})
    merge_results(JSON, {"ablation_text_bm25": out})


def dense_ablation() -> None:
    """Dense text-variant and query-prefix ablation, on the dev queries' products only."""
    from cartrank.search.dense import embed_corpus

    ev = SearchEval("dev")
    dev_ids = sorted({p for q in ev.qrels.values() for p in q})
    enc = load_encoder(BASE_MODEL)
    ids = ev.ids
    keep = [i for i, p in enumerate(ids) if p in set(dev_ids)]
    out = {}
    from cartrank.eval.metrics import evaluate
    from cartrank.eval.search_eval import rank_by_scores

    for v in VARIANTS:
        embed_corpus(f"abl_{BASE_MODEL.split('/')[-1]}", BASE_MODEL, v, ids=dev_ids)
        vecs = np.load(
            Path(__file__).resolve().parents[3]
            / "artifacts/dense"
            / f"abl_{BASE_MODEL.split('/')[-1]}_{v}.npy"
        )
        row = {ids[i]: j for j, i in enumerate(keep)}
        for prefix_name, prefix in (("prefix", None), ("no_prefix", "")):
            if v != "tbcb" and prefix_name == "no_prefix":
                continue
            kw = {} if prefix is None else {"prefix": prefix}
            qv = encode_queries(enc, [ev.queries[q] for q in ev.qids], **kw)
            rr = {}
            for i, q in enumerate(ev.qids):
                judged = list(ev.qrels[q])
                rr[q] = rank_by_scores(judged, vecs[[row[p] for p in judged]] @ qv[i])
            out[f"{v}|{prefix_name}"] = evaluate(rr, ev.qrels)
            print(v, prefix_name, round(out[f"{v}|{prefix_name}"]["ndcg@10"], 4))
    merge_results(JSON, {"ablation_text_dense": out})


def main_ladder() -> None:
    ev = SearchEval("dev")
    ev.run_bm25()
    ev.run_dense("dense_base", "base")
    # tune RRF k and per-list depth on dev (full-corpus NDCG, the harder setting)
    tuning = []
    for k in (10, 30, 60, 100):
        for depth in (50, 100, 200):
            ev.run_hybrid("hybrid_base", ("bm25", "dense_base"), k=k, depth=depth)
            m = ev.score("hybrid_base", "full")
            tuning.append({"k": k, "depth": depth, **m})
    best = max(tuning, key=lambda r: r["ndcg@10"])
    ev.run_hybrid("hybrid_base", ("bm25", "dense_base"), k=best["k"], depth=best["depth"])
    ladder = _load().get("ladder", {})
    for name, _ in LADDER:
        for mode in ("rerank", "full"):
            ladder[f"{name}|{mode}"] = ev.score(name, mode)
    # latency of the full-corpus stage, one query at a time, local CPU/MPS, 200 dev queries
    sample = [ev.queries[q] for q in ev.qids[:200]]
    enc = load_encoder(BASE_MODEL)
    index = ev.dense["dense_base"][1]

    def dense_q(q):
        index.search(encode_queries(enc, [q]), 100)

    lat = {
        "bm25": latency_ms(lambda q: ev.bm25.search(q, 100), sample),
        "dense_base": latency_ms(dense_q, sample),
    }
    lat["hybrid_base"] = latency_ms(
        lambda q: rrf_both(ev, enc, index, q, best["k"], best["depth"]), sample
    )
    vecs = ev.dense["dense_base"][0]
    with tempfile.TemporaryDirectory() as d:
        ev.bm25.retriever.save(d)
        bm25_mb = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e6
    index_info = {
        "bm25_build_seconds": ev.bm25.build_seconds,
        "bm25_index_mb": bm25_mb,
        "dense_vectors_mb": vecs.nbytes / 1e6,
        "dense_dim": int(vecs.shape[1]),
        "corpus_size": len(ev.ids),
    }
    merge_results(
        JSON,
        {
            "ladder": ladder,
            "hybrid_tuning": tuning,
            "hybrid_best": best,
            "latency_ms": lat,
            "index": index_info,
        },
    )
    print(json.dumps({k: ladder[k] for k in ladder}, indent=1))


def rrf_both(ev, enc, index, q, k, depth):
    from cartrank.search.hybrid import rrf

    a = ev.bm25.search(q, depth)
    _, top = index.search(encode_queries(enc, [q]), depth)
    return rrf([a, [ev.ids[j] for j in top[0]]], k=k, depth=depth)


def report() -> None:
    from cartrank.eval.report_search import write_report

    write_report()


if __name__ == "__main__":
    {
        "bm25-ablation": bm25_ablation,
        "dense-ablation": dense_ablation,
        "main": main_ladder,
        "report": report,
    }[sys.argv[1]]()
