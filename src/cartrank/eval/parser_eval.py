"""Parser evaluation against evals/queries/parser_gold.jsonl.

Scored fields: route, price_min, price_max, brand (case-insensitive), min_rating, must_have and
exclude (as sets of lower-cased strings). `text` is scored as mean token F1 (free-form English
phrase, so exact match is too brittle). `category` is not scored: it has no canonical vocabulary.
"Constraint intent" exact match = all scored fields except text.
"""

import re
from collections import defaultdict

import numpy as np

from cartrank.query.schema import QueryIntent

EXACT_FIELDS = ["route", "price_min", "price_max", "brand", "min_rating", "must_have", "exclude"]


def _norm_str(s: str | None) -> str | None:
    return None if s is None else re.sub(r"\s+", " ", s.replace("-", " ").strip().lower())


def _norm(field: str, v):
    if field in ("must_have", "exclude"):
        return frozenset(_norm_str(x) for x in v)
    if field == "brand":
        return _norm_str(v)
    return v


def field_match(field: str, gold: QueryIntent, pred: QueryIntent) -> bool:
    return _norm(field, getattr(gold, field)) == _norm(field, getattr(pred, field))


def token_f1(a: str, b: str) -> float:
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not ta and not tb:
        return 1.0
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def score(rows: list[dict]) -> dict:
    """rows: [{"group", "gold", "pred"}] (QueryIntents). Returns metrics overall and by group."""

    def summarize(sub: list[dict]) -> dict:
        out = {"n": len(sub)}
        for f in EXACT_FIELDS:
            out[f] = float(np.mean([field_match(f, r["gold"], r["pred"]) for r in sub]))
        out["text_f1"] = float(np.mean([token_f1(r["gold"].text, r["pred"].text) for r in sub]))
        out["constraint_intent_exact"] = float(
            np.mean([all(field_match(f, r["gold"], r["pred"]) for f in EXACT_FIELDS) for r in sub])
        )
        return out

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    return {"overall": summarize(rows), "by_group": {g: summarize(v) for g, v in by_group.items()}}
