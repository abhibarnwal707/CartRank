"""BM25 + dense (+ optional cross-encoder) search over the Office_Products demo catalog.

Dense search is a plain numpy dot product (51K x 384), so faiss is not needed here and cannot
clash with CPU torch (see the macOS note in CLAUDE.md).
"""

import json

import numpy as np
import pandas as pd

from cartrank.data.catalog import CATALOG_PATH
from cartrank.data.paths import RESULTS, ROOT
from cartrank.search.bm25 import Bm25Index
from cartrank.search.dense import BGE_QUERY_PREFIX, encode_queries, load_encoder
from cartrank.search.hybrid import rrf
from cartrank.search.rerank import load_cross_encoder, rerank

CATALOG_ARTIFACTS = ROOT / "artifacts" / "catalog"


class CatalogSearch:
    def __init__(self, tag: str, model_path: str, hybrid_k: int, hybrid_depth: int):
        self.cat = pd.read_parquet(CATALOG_PATH)
        self.ids = self.cat.parent_asin.tolist()
        self.text = dict(zip(self.ids, self.cat.text, strict=True))
        self.bm25 = Bm25Index(self.ids, self.cat.text.tolist())
        self.enc = load_encoder(model_path)
        vec_path = CATALOG_ARTIFACTS / f"{tag}_vecs.npy"
        if vec_path.exists():
            self.vecs = np.load(vec_path)
        else:
            self.vecs = self.enc.encode(
                self.cat.text.tolist(),
                batch_size=128,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            CATALOG_ARTIFACTS.mkdir(parents=True, exist_ok=True)
            np.save(vec_path, self.vecs)
        self.k, self.depth = hybrid_k, hybrid_depth
        self.ce = None

    def dense(self, query: str, n: int) -> list[str]:
        q = encode_queries(self.enc, [query], prefix=BGE_QUERY_PREFIX)[0]
        s = self.vecs @ q
        top = np.argpartition(-s, n)[:n]
        return [self.ids[i] for i in top[np.argsort(-s[top], kind="stable")]]

    def hybrid(self, query: str) -> list[str]:
        return rrf(
            [self.bm25.search(query, self.depth), self.dense(query, self.depth)],
            k=self.k,
            depth=self.depth,
        )

    def final(self, query: str, ce_k: int = 50) -> list[str]:
        """Hybrid top candidates, with the first ce_k re-ordered by the cross-encoder."""
        if self.ce is None:
            self.ce = load_cross_encoder()
        cands = self.hybrid(query)[:100]
        head = cands[:ce_k]
        scores = self.ce.predict([(query, self.text[p]) for p in head], batch_size=64)
        sc = {(0, p): float(s) for p, s in zip(head, scores, strict=True)}
        return rerank(0, cands, sc, ce_k)


def hybrid_setting(tag: str) -> tuple[int, int]:
    d = json.loads((RESULTS / "search_dev.json").read_text())
    best = d["hybrid_best"] if tag == "base" else d[f"hybrid_best_{tag}"]
    return best["k"], best["depth"]
