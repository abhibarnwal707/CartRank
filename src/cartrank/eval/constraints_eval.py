"""End-to-end constraint satisfaction: plain hybrid search vs parser + hard filters.

Constraints come from the GOLD labels (what the user stated), never from the parser. A product
satisfies a constraint by a deterministic rule on catalog fields:
  price / rating : objective metadata (price, average_rating)
  brand          : normalised brand string appears in the store or title
  must_have      : the term appears in title or features (not negated there)
  exclude        : the term does not appear in title or features (unless negated there)
Price, rating and brand are objective; must_have / exclude are keyword rules, not human
judgements, and the filter applies the same rules, so those two are a mechanism check only.
"""

import re

from cartrank.query.schema import QueryIntent

NEGATORS = {"no", "without", "non", "not", "free"}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace("-", " "))


def mentions(text: str, term: str) -> bool:
    """True if `term` appears in `text` at least once without a nearby negation."""
    toks, t = _tokens(text), _tokens(term)
    if not t:
        return False
    n = len(t)
    for i in range(len(toks) - n + 1):
        if toks[i : i + n] == t:
            before = toks[max(0, i - 2) : i]
            after = toks[i + n : i + n + 1]
            if not (NEGATORS & set(before)) and "free" not in after:
                return True
    return False


def check(item: dict, c: QueryIntent) -> dict[str, bool | None]:
    """Per-constraint result: True/False, or None when the query states no such constraint."""
    blob = f"{item['title']} {' '.join(item['features'])}"
    out: dict[str, bool | None] = {
        "price": None,
        "rating": None,
        "brand": None,
        "must_have": None,
        "exclude": None,
    }
    if c.price_min is not None or c.price_max is not None:
        p = item["price"]
        out["price"] = bool(
            p is not None
            and (c.price_min is None or p >= c.price_min)
            and (c.price_max is None or p <= c.price_max)
        )
    if c.min_rating is not None:
        out["rating"] = bool(item["average_rating"] >= c.min_rating)
    if c.brand:
        b = " ".join(_tokens(c.brand))
        out["brand"] = b in " ".join(_tokens(f"{item['store'] or ''} {item['title']}"))
    if c.must_have:
        out["must_have"] = all(mentions(blob, m) for m in c.must_have)
    if c.exclude:
        out["exclude"] = not any(mentions(blob, e) for e in c.exclude)
    return out


def satisfies_all(item: dict, c: QueryIntent, kinds: tuple[str, ...] | None = None) -> bool:
    res = check(item, c)
    return all(v for k, v in res.items() if v is not None and (kinds is None or k in kinds))


def n_constraints(c: QueryIntent, kinds: tuple[str, ...] | None = None) -> int:
    res = check({"title": "", "features": [], "price": 0, "average_rating": 0, "store": ""}, c)
    n = 0
    for k, v in res.items():
        if v is None or (kinds is not None and k not in kinds):
            continue
        n += len(c.must_have) if k == "must_have" else len(c.exclude) if k == "exclude" else 1
    return n


OBJECTIVE = ("price", "rating", "brand")
