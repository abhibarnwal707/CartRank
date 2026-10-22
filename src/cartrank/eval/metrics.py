"""Ranking metrics on ESCI graded labels.

Gains follow the ESCI repo's trec_eval call (`ndcg.1=0,2=0.01,3=0.1,4=1`):
E = 1.0, S = 0.1, C = 0.01, I = 0.
"""

import math
from collections.abc import Mapping, Sequence

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: Sequence[str], labels: Mapping[str, str], k: int = 10) -> float:
    """`ranked` = product ids best first; `labels` = judged product id -> E/S/C/I.

    Unjudged products count as gain 0. The ideal ranking sorts all judged gains.
    """
    gains = [GAINS[labels[p]] if p in labels else 0.0 for p in ranked]
    ideal = sorted((GAINS[v] for v in labels.values()), reverse=True)
    best = dcg(ideal, k)
    return dcg(gains, k) / best if best > 0 else 0.0


def recall_at_k(ranked: Sequence[str], labels: Mapping[str, str], k: int = 100) -> float | None:
    """Fraction of the query's Exact products found in the top k.

    Returns None when the query has no Exact product (excluded from the mean).
    """
    exact = {p for p, v in labels.items() if v == "E"}
    if not exact:
        return None
    return len(exact & set(ranked[:k])) / len(exact)


def mrr_at_k(ranked: Sequence[str], labels: Mapping[str, str], k: int = 10) -> float:
    """Reciprocal rank of the first Exact product within the top k (0 if none)."""
    for i, p in enumerate(ranked[:k]):
        if labels.get(p) == "E":
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    rankings: Mapping[int, Sequence[str]], qrels: Mapping[int, Mapping[str, str]]
) -> dict[str, float]:
    """Mean NDCG@10, MRR@10 over all queries; Recall@100 over queries with >= 1 Exact."""
    qids = list(qrels)
    ndcg = [ndcg_at_k(rankings.get(q, []), qrels[q], 10) for q in qids]
    mrr = [mrr_at_k(rankings.get(q, []), qrels[q], 10) for q in qids]
    rec = [r for q in qids if (r := recall_at_k(rankings.get(q, []), qrels[q], 100)) is not None]
    return {
        "ndcg@10": sum(ndcg) / len(ndcg),
        "recall@100": sum(rec) / len(rec),
        "mrr@10": sum(mrr) / len(mrr),
        "n_queries": len(qids),
    }
