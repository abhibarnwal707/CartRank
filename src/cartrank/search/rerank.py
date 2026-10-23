"""Cross-encoder reranking with a persistent (query, product) -> score cache."""

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd

from cartrank.data.paths import ROOT

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LENGTH = 256
CACHE_DIR = ROOT / "artifacts" / "rerank"


def load_cross_encoder(model: str = CE_MODEL):
    from sentence_transformers import CrossEncoder

    # CPU on purpose: the demo runs on CPU, so latency is measured on CPU.
    return CrossEncoder(model, device="cpu", max_length=CE_MAX_LENGTH)


class ScoreCache:
    """Scores keyed by (query_id, product_id); lets runs resume and share pairs across k."""

    def __init__(self, tag: str):
        self.path = CACHE_DIR / f"ce_{tag}.parquet"
        self.scores: dict[tuple[int, str], float] = {}
        if self.path.exists():
            df = pd.read_parquet(self.path)
            self.scores = {
                (int(q), p): s for q, p, s in zip(df.query_id, df.product_id, df.score, strict=True)
            }

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rows = [(q, p, s) for (q, p), s in self.scores.items()]
        pd.DataFrame(rows, columns=["query_id", "product_id", "score"]).to_parquet(self.path)

    def fill(
        self,
        model,
        pairs: Iterable[tuple[int, str]],
        query_text: dict[int, str],
        product_text: dict[str, str],
        batch_size: int = 64,
        chunk: int = 20_000,
        log=print,
    ) -> None:
        todo = [pr for pr in dict.fromkeys(pairs) if pr not in self.scores]
        t0 = time.time()
        for start in range(0, len(todo), chunk):
            part = todo[start : start + chunk]
            inputs = [(query_text[q], product_text[p]) for q, p in part]
            out = model.predict(inputs, batch_size=batch_size, show_progress_bar=False)
            self.scores.update(zip(part, np.asarray(out, dtype=np.float32).tolist(), strict=True))
            self.save()
            log(f"scored {start + len(part)}/{len(todo)} pairs ({time.time() - t0:.0f}s)")


def rerank(
    qid: int, candidates: list[str], scores: dict[tuple[int, str], float], k: int
) -> list[str]:
    """Re-order the first k candidates by cross-encoder score; the tail keeps its order."""
    head = candidates[:k]
    head = sorted(head, key=lambda p: -scores[(qid, p)])
    return head + candidates[k:]
