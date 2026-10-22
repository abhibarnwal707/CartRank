from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, depth: int | None = None) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked[:depth], start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])
