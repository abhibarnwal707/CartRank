"""BM25 over the product corpus using bm25s (sparse, CPU)."""

import time

import bm25s
import numpy as np
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


class Bm25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = np.array(ids)
        self.id_to_row = {p: i for i, p in enumerate(ids)}
        t0 = time.time()
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)
        self.build_seconds = time.time() - t0

    def _query_tokens(self, query: str) -> list[str]:
        toks = bm25s.tokenize(
            query, stopwords="en", stemmer=_STEMMER, return_ids=False, show_progress=False
        )
        return toks[0]

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every corpus document for one query."""
        toks = self._query_tokens(query)
        if not toks:
            return np.zeros(len(self.ids), dtype=np.float32)
        return self.retriever.get_scores(toks)

    def search(self, query: str, k: int) -> list[str]:
        s = self.scores(query)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return self.ids[top].tolist()
