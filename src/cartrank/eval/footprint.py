"""Serving footprint and per-stage latency of the catalog search, on CPU (the demo has no GPU).

python -m cartrank.eval.footprint <tag> <encoder_path>  -> results/serving_footprint.md

Measured on the local machine (macOS, Apple Silicon). Streamlit Cloud (Linux) numbers are
measured separately in Phase 6; resident memory differs between operating systems.
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cartrank.data.catalog import CATALOG_PATH
from cartrank.data.paths import RESULTS, ROOT
from cartrank.eval.search_eval import latency_ms
from cartrank.search.catalog_index import CATALOG_ARTIFACTS, hybrid_setting
from cartrank.search.hybrid import rrf

QUERIES = ROOT / "evals" / "queries" / "demo_queries.jsonl"
BUDGET_MB = 2000


def rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.strip()) / 1024


def main(tag: str, model_path: str) -> None:
    from cartrank.search.bm25 import Bm25Index
    from cartrank.search.dense import BGE_QUERY_PREFIX, encode_queries, load_encoder
    from cartrank.search.rerank import load_cross_encoder

    steps: list[tuple[str, float]] = [("Python + libraries imported", rss_mb())]
    t_start = time.time()

    cat = pd.read_parquet(CATALOG_PATH)
    ids, texts = cat.parent_asin.tolist(), cat.text.tolist()
    text = dict(zip(ids, texts, strict=True))
    steps.append(("Catalog metadata (51,814 items)", rss_mb()))

    bm25 = Bm25Index(ids, texts)
    steps.append(("BM25 index", rss_mb()))

    vecs = np.load(CATALOG_ARTIFACTS / f"{tag}_vecs.npy")
    steps.append(("Dense vectors (float32, 384-dim)", rss_mb()))

    enc = load_encoder(model_path, device="cpu")
    steps.append(("Fine-tuned bi-encoder (CPU)", rss_mb()))

    ce = load_cross_encoder()
    steps.append(("Cross-encoder MiniLM-L6 (CPU)", rss_mb()))
    cold_start = time.time() - t_start

    k, depth = hybrid_setting(tag)
    queries = [json.loads(line)["query"] for line in QUERIES.read_text().splitlines()]

    def dense(q: str, n: int) -> list[str]:
        v = encode_queries(enc, [q], prefix=BGE_QUERY_PREFIX)[0]
        s = vecs @ v
        top = np.argpartition(-s, n)[:n]
        return [ids[i] for i in top[np.argsort(-s[top], kind="stable")]]

    def full(q: str) -> list[str]:
        cands = rrf([bm25.search(q, depth), dense(q, depth)], k=k, depth=depth)[:100]
        head = cands[:50]
        ce.predict([(q, text[p]) for p in head], batch_size=64)
        return cands

    stages = {
        "BM25 top-200": latency_ms(lambda q: bm25.search(q, depth), queries),
        "Dense (encode query + search)": latency_ms(lambda q: dense(q, depth), queries),
        "Hybrid retrieve (BM25 + dense + RRF)": latency_ms(
            lambda q: rrf([bm25.search(q, depth), dense(q, depth)], k=k, depth=depth), queries
        ),
        "Cross-encoder rerank, top-50": latency_ms(
            lambda q: ce.predict(
                [(q, text[p]) for p in rrf([bm25.search(q, 50)], k=k)[:50]], batch_size=64
            ),
            queries,
        ),
        "Whole pipeline (retrieve + rerank)": latency_ms(full, queries),
    }
    after_queries = rss_mb()
    disk = {
        "catalog.parquet": CATALOG_PATH.stat().st_size / 1e6,
        f"{tag}_vecs.npy": (CATALOG_ARTIFACTS / f"{tag}_vecs.npy").stat().st_size / 1e6,
    }

    prev = 0.0
    rows = []
    for name, r in steps:
        rows.append(f"| {name} | {r:,.0f} | {r - prev:+,.0f} |")
        prev = r
    md = (
        "# Serving footprint (demo catalog, CPU)\n\n"
        f"Measured locally on {platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}, encoder and cross-encoder on CPU. "
        "**Not yet measured on Streamlit Cloud** (Linux); that is Phase 6. "
        "Resident set size (RSS) via `ps`, cumulative as each component loads.\n\n"
        "| Step | RSS MB | Added MB |\n|---|---|---|\n"
        + "\n".join(rows)
        + f"\n| After running {len(queries)} queries | {after_queries:,.0f} | "
        f"{after_queries - prev:+,.0f} |\n\n"
        f"Budget: {BUDGET_MB:,} MB. Final steady-state RSS {after_queries:,.0f} MB "
        f"({'within' if after_queries <= BUDGET_MB else 'OVER'} budget).\n\n"
        f"Component load time (catalog, BM25 build, vectors, both models; models already "
        f"cached on disk): {cold_start:.0f} s. This is not the full app cold start, which also "
        "includes downloading artifacts and starting Streamlit.\n\n"
        "## Per-stage latency (ms, one query at a time, 50 demo queries, CPU)\n\n"
        "| Stage | p50 | p95 |\n|---|---|---|\n"
        + "\n".join(f"| {n} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} |" for n, v in stages.items())
        + "\n\nThe cross-encoder row re-ranks BM25's top 50 only (a timing proxy for 50 pairs); "
        "the whole-pipeline row is the real hybrid top-50 rerank.\n\n"
        "## Artifact sizes on disk (MB)\n\n"
        + "\n".join(f"- {n}: {v:,.0f}" for n, v in disk.items())
        + "\n"
    )
    (RESULTS / "serving_footprint.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
