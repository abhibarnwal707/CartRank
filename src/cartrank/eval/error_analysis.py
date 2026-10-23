"""Phase 2.6: dev queries where the final system (hybrid-FT + cross-encoder) loses to BM25.

Full-corpus mode. Writes results/search_error_analysis_raw.json: per-query details for a seeded
sample of 30 losing queries, to be read and labelled by hand into search_error_analysis.md.
"""

import json
import re

import numpy as np

from cartrank.data.paths import RESULTS
from cartrank.eval.metrics import ndcg_at_k
from cartrank.eval.search_eval import SearchEval
from cartrank.search.rerank import CACHE_DIR, ScoreCache, rerank

NEGATION = re.compile(r"\b(no|not|without|non|free|except|excluding|under|less)\b|-\w+", re.I)
K = 50


def query_tags(query: str, brands: set[str]) -> list[str]:
    toks = query.lower().split()
    tags = []
    if len(toks) <= 2:
        tags.append("short(<=2 tokens)")
    if len(toks) >= 8:
        tags.append("long(>=8 tokens)")
    if re.search(r"\d", query):
        tags.append("has_number")
    if NEGATION.search(query):
        tags.append("negation_or_exclusion")
    if any(t in brands for t in toks):
        tags.append("brand_in_query")
    return tags


def main(tag: str = "ft", sample_size: int = 30, seed: int = 0) -> None:
    ev = SearchEval("dev")
    ev.run_bm25()
    cands = {
        int(q): c
        for q, c in json.loads((CACHE_DIR / f"cand_hybrid_{tag}.json").read_text()).items()
    }
    sc = ScoreCache("minilm_l6_tbcb").scores
    final = {q: rerank(q, cands[q], sc, K) for q in ev.qids}

    import pandas as pd

    brands = {
        b.lower()
        for b in pd.read_parquet(
            "data/esci/shopping_queries_dataset_products.parquet", columns=["product_brand"]
        )
        .product_brand.dropna()
        .unique()
        if len(b) > 2
    }
    title = dict(zip(ev.ids, [t.split(". ")[0] for t in ev.texts], strict=True))

    rows = []
    for q in ev.qids:
        b = ndcg_at_k(ev.full_lists["bm25"][q], ev.qrels[q], 10)
        f = ndcg_at_k(final[q], ev.qrels[q], 10)
        if f < b:
            rows.append((q, b, f))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(sample_size, len(rows)), replace=False)

    def top(lst, q):
        return [{"label": ev.qrels[q].get(p, "unjudged"), "title": title[p][:90]} for p in lst[:5]]

    out = {
        "n_dev_queries": len(ev.qids),
        "n_losing": len(rows),
        "pct_losing": len(rows) / len(ev.qids),
        "samples": [],
    }
    for i in pick:
        q, b, f = rows[i]
        bm = ev.full_lists["bm25"][q][:10]
        fi = final[q][:10]
        out["samples"].append(
            {
                "query_id": q,
                "query": ev.queries[q],
                "tags": query_tags(ev.queries[q], brands),
                "ndcg_bm25": round(b, 4),
                "ndcg_final": round(f, 4),
                "unjudged_top10_bm25": sum(p not in ev.qrels[q] for p in bm),
                "unjudged_top10_final": sum(p not in ev.qrels[q] for p in fi),
                "bm25_top5": top(bm, q),
                "final_top5": top(fi, q),
            }
        )
    (RESULTS / "search_error_analysis_raw.json").write_text(json.dumps(out, indent=1))
    print(out["n_losing"], "losing of", out["n_dev_queries"])


if __name__ == "__main__":
    main()
