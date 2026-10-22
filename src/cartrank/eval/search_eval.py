"""Search evaluation on ESCI: rerank mode and full-corpus retrieval mode (never mixed).

Rerank mode: rank only each query's judged products (every candidate has a label).
Full-corpus mode: retrieve from all US small-set products; unjudged products count as
non-relevant, which underestimates scores.
"""

import json
import time

import numpy as np
import pandas as pd

from cartrank.data.paths import PROCESSED, RESULTS
from cartrank.eval.metrics import evaluate
from cartrank.search.bm25 import Bm25Index
from cartrank.search.corpus import DEFAULT_VARIANT, load_corpus
from cartrank.search.dense import (
    BASE_MODEL,
    BGE_QUERY_PREFIX,
    encode_queries,
    flat_ip_index,
    load_encoder,
    vectors_path,
)
from cartrank.search.hybrid import rrf

DEPTH = 200  # candidates kept per full-corpus query


def load_split(name: str) -> tuple[dict[int, str], dict[int, dict[str, str]]]:
    df = pd.read_parquet(PROCESSED / "esci" / f"{name}.parquet")
    queries = df.drop_duplicates("query_id").set_index("query_id")["query"].to_dict()
    qrels = {
        int(q): dict(zip(g.product_id, g.esci_label, strict=True))
        for q, g in df.groupby("query_id")
    }
    return queries, qrels


def rank_by_scores(pids: list[str], scores: np.ndarray) -> list[str]:
    return [pids[i] for i in np.argsort(-scores, kind="stable")]


class SearchEval:
    """Holds one split's queries, the corpus and every system's candidate lists."""

    def __init__(self, split: str, variant: str = DEFAULT_VARIANT):
        self.split = split
        self.queries, self.qrels = load_split(split)
        self.qids = list(self.queries)
        corpus = load_corpus()
        self.ids = corpus.product_id.tolist()
        self.row = {p: i for i, p in enumerate(self.ids)}
        self.variant = variant
        self.texts = corpus[f"text_{variant}"].tolist()
        self.bm25: Bm25Index | None = None
        self.dense: dict[str, tuple[np.ndarray, object]] = {}
        # per system: qid -> list of product ids (best first)
        self.rerank_lists: dict[str, dict[int, list[str]]] = {}
        self.full_lists: dict[str, dict[int, list[str]]] = {}
        self.encoder = None

    # ---- BM25 -------------------------------------------------------------------------
    def run_bm25(self, name: str = "bm25") -> None:
        if self.bm25 is None:
            self.bm25 = Bm25Index(self.ids, self.texts)
        rr, full = {}, {}
        for q in self.qids:
            s = self.bm25.scores(self.queries[q])
            judged = list(self.qrels[q])
            rr[q] = rank_by_scores(judged, s[[self.row[p] for p in judged]])
            top = np.argpartition(-s, DEPTH)[:DEPTH]
            full[q] = [self.ids[i] for i in top[np.argsort(-s[top], kind="stable")]]
        self.rerank_lists[name], self.full_lists[name] = rr, full

    # ---- Dense ------------------------------------------------------------------------
    def run_dense(
        self, name: str, tag: str, model: str = BASE_MODEL, prefix: str = BGE_QUERY_PREFIX
    ) -> None:
        if self.encoder is None or self.encoder[0] != model:
            self.encoder = (model, load_encoder(model))
        enc = self.encoder[1]
        vecs = np.load(vectors_path(tag, self.variant))
        index = flat_ip_index(vecs)
        self.dense[name] = (vecs, index)
        qvecs = encode_queries(enc, [self.queries[q] for q in self.qids], prefix=prefix)
        _, top = index.search(qvecs, DEPTH)
        rr, full = {}, {}
        for i, q in enumerate(self.qids):
            judged = list(self.qrels[q])
            s = vecs[[self.row[p] for p in judged]] @ qvecs[i]
            rr[q] = rank_by_scores(judged, s)
            full[q] = [self.ids[j] for j in top[i]]
        self.rerank_lists[name], self.full_lists[name] = rr, full

    # ---- Hybrid -----------------------------------------------------------------------
    def run_hybrid(self, name: str, parts: tuple[str, str], k: int = 60, depth: int = 100) -> None:
        a, b = parts
        self.rerank_lists[name] = {
            q: rrf([self.rerank_lists[a][q], self.rerank_lists[b][q]], k=k) for q in self.qids
        }
        self.full_lists[name] = {
            q: rrf([self.full_lists[a][q], self.full_lists[b][q]], k=k, depth=depth)
            for q in self.qids
        }

    # ---- Scoring ----------------------------------------------------------------------
    def score(self, name: str, mode: str) -> dict[str, float]:
        lists = self.rerank_lists if mode == "rerank" else self.full_lists
        return evaluate(lists[name], self.qrels)


def latency_ms(fn, items, warmup: int = 5) -> dict[str, float]:
    for x in items[:warmup]:
        fn(x)
    times = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def merge_results(path_name: str, new: dict) -> None:
    """Merge `new` into results/<path_name> (a JSON file) so later phases can add ladder rows."""
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / path_name
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(new)
    p.write_text(json.dumps(cur, indent=2))
