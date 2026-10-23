"""Phase 2 driver: python -m cartrank.eval.run_phase2 <rerank|report> [first_stage_tag]

Cross-encoder rerank on ESCI dev. `rerank` uses the base (off-the-shelf) hybrid as first stage
unless a fine-tuned encoder tag is given (see docs/kaggle_finetune.md).
"""

import json
import sys

import numpy as np

from cartrank.data.paths import RESULTS
from cartrank.eval.metrics import evaluate
from cartrank.eval.search_eval import SearchEval, latency_ms, merge_results, rank_by_scores
from cartrank.search.dense import BASE_MODEL
from cartrank.search.rerank import CACHE_DIR, CE_MODEL, ScoreCache, load_cross_encoder, rerank

JSON = "search_dev.json"
KS = (20, 50, 100)


def _ladder() -> dict:
    p = RESULTS / JSON
    return json.loads(p.read_text()).get("ladder", {}) if p.exists() else {}


def _cand_path(tag: str):
    return CACHE_DIR / f"cand_hybrid_{tag}.json"


def run_candidates(tag: str = "base", model: str | None = None) -> None:
    """Stage 1 (faiss + MPS): hybrid top-100 per dev query. Kept in its own process because
    faiss and CPU torch inference in one process deadlock on macOS.

    For the base encoder the stored Phase 1 hybrid setting is reused; for a fine-tuned encoder
    (`tag` != "base") RRF k and depth are re-tuned on dev, and dense/hybrid ladder rows are added.
    """
    import faiss  # noqa: F401  # must load before torch (libomp order), else segfault

    ev = SearchEval("dev")
    ev.run_bm25()
    dense, first = f"dense_{tag}", f"hybrid_{tag}"
    ev.run_dense(dense, tag, model=model or BASE_MODEL)
    stored = json.loads((RESULTS / JSON).read_text())
    if tag == "base":
        best = stored["hybrid_best"]
    else:
        tuning = []
        for k in (10, 30, 60, 100):
            for depth in (50, 100, 200):
                ev.run_hybrid(first, ("bm25", dense), k=k, depth=depth)
                tuning.append({"k": k, "depth": depth, **ev.score(first, "full")})
        best = max(tuning, key=lambda r: r["ndcg@10"])
    ev.run_hybrid(first, ("bm25", dense), k=best["k"], depth=best["depth"])
    if tag != "base":
        ladder = stored.get("ladder", {})
        for name in (dense, first):
            for mode in ("rerank", "full"):
                ladder[f"{name}|{mode}"] = ev.score(name, mode)
        merge_results(
            JSON,
            {"ladder": ladder, f"hybrid_tuning_{tag}": tuning, f"hybrid_best_{tag}": best},
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cands = {str(q): ev.full_lists[first][q][: max(KS)] for q in ev.qids}
    _cand_path(tag).write_text(json.dumps(cands))
    print("wrote", _cand_path(tag), len(cands), "best", best)


def run_rerank(tag: str = "base") -> None:
    ev = SearchEval("dev")
    first = f"hybrid_{tag}"
    ev.full_lists[first] = {int(q): c for q, c in json.loads(_cand_path(tag).read_text()).items()}

    corpus_text = dict(zip(ev.ids, ev.texts, strict=True))
    cache = ScoreCache("minilm_l6_tbcb")
    model = load_cross_encoder()
    judged = [(q, p) for q in ev.qids for p in ev.qrels[q]]
    top = [(q, p) for q in ev.qids for p in ev.full_lists[first][q]]
    cache.fill(model, judged + top, ev.queries, corpus_text)
    sc = cache.scores

    ladder = _ladder()
    # rerank mode: the cross-encoder orders every judged candidate
    rr = {
        q: rank_by_scores(list(ev.qrels[q]), np.array([sc[(q, p)] for p in ev.qrels[q]]))
        for q in ev.qids
    }
    ladder[f"{first}_ce|rerank"] = evaluate(rr, ev.qrels)
    tuning = []
    for k in KS:
        full = {q: rerank(q, ev.full_lists[first][q], sc, k) for q in ev.qids}
        m = evaluate(full, ev.qrels)
        ladder[f"{first}_ce_k{k}|full"] = m
        tuning.append({"k": k, **m})
        print(k, round(m["ndcg@10"], 4))

    # CPU latency of the cross-encoder stage alone, 50 dev queries, first-stage top-k
    lat = {}
    for k in KS:
        sample = [
            [(ev.queries[q], corpus_text[p]) for p in ev.full_lists[first][q][:k]]
            for q in ev.qids[:50]
        ]
        lat[f"k={k}"] = latency_ms(lambda x: model.predict(x, batch_size=64), sample, warmup=3)
    n_params = sum(p.numel() for p in model.model.parameters())
    merge_results(
        JSON,
        {
            "ladder": ladder,
            f"rerank_tuning_{tag}": tuning,
            f"rerank_latency_ms_{tag}": lat,
            "rerank_model": {"name": CE_MODEL, "params": n_params, "max_length": 256},
        },
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "candidates":
        run_candidates(
            sys.argv[2] if len(sys.argv) > 2 else "base",
            sys.argv[3] if len(sys.argv) > 3 else None,
        )
    if cmd == "rerank":
        run_rerank(sys.argv[2] if len(sys.argv) > 2 else "base")
