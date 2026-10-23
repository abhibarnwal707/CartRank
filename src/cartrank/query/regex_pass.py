"""Deterministic pre-pass: price ranges and minimum ratings. Wins over the LLM on conflict."""

import re
from dataclasses import dataclass

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
CUR_BEFORE = r"(?:\$\s*|usd\s*|us\$\s*)"
CUR_AFTER = r"(?:\s*(?:dollars?|bucks|usd)\b)"
# a number followed by one of these is a quantity or size, not a price
UNIT = (
    r"(?!\s*(?:inch(?:es)?|in\b|mm\b|cm\b|ft\b|feet\b|oz\b|lbs?\b|packs?\b|ct\b|count\b|pcs\b|"
    r"pieces?\b|sheets?\b|pages?\b|gb\b|tb\b|mb\b|ml\b|gal(?:lons?)?\b|sets?\b|x\b|%|stars?\b|"
    r"yards?\b|pens?\b|pack\b))"
)

MAX_WORDS = (
    r"(?:under|below|less than|cheaper than|up to|at most|no more than|not more than|"
    r"max(?:imum)?(?: price)?(?: of)?|within|budget of)"
)
MIN_WORDS = r"(?:over|above|more than|at least|min(?:imum)?(?: price)?(?: of)?|starting at|from)"

_FLAGS = re.IGNORECASE


def _num(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class RegexResult:
    price_min: float | None = None
    price_max: float | None = None
    min_rating: float | None = None
    text: str = ""


def _has_cur(span: str) -> bool:
    return bool(re.search(CUR_BEFORE + "|" + CUR_AFTER, span, _FLAGS))


def parse(query: str) -> RegexResult:
    res = RegexResult()
    spans: list[tuple[int, int]] = []

    def claim(m: re.Match) -> bool:
        if any(m.start() < e and s < m.end() for s, e in spans):
            return False
        spans.append((m.start(), m.end()))
        return True

    # 1. ranges: "between 20 and 40", "$20-$40", "20 to 40 dollars"
    range_re = re.compile(
        rf"(?:between\s+)?(?P<c1>{CUR_BEFORE})?(?P<a>{NUM})(?P<a_after>{CUR_AFTER})?\s*"
        rf"(?:and|to|-|–)\s*(?P<c2>{CUR_BEFORE})?(?P<b>{NUM})(?P<b_after>{CUR_AFTER})?{UNIT}",
        _FLAGS,
    )
    for m in range_re.finditer(query):
        explicit_between = m.group(0).lower().startswith("between")
        if not (_has_cur(m.group(0)) or explicit_between):
            continue
        lo, hi = sorted((_num(m.group("a")), _num(m.group("b"))))
        if claim(m):
            res.price_min, res.price_max = lo, hi

    # 2. maximum: "under $50", "less than 20 dollars", "below 15"
    max_re = re.compile(
        rf"\b{MAX_WORDS}\s+(?P<c>{CUR_BEFORE})?(?P<n>{NUM})(?P<after>{CUR_AFTER})?{UNIT}", _FLAGS
    )
    for m in max_re.finditer(query):
        if claim(m) and res.price_max is None:
            res.price_max = _num(m.group("n"))

    # 3. minimum: needs a currency marker ("at least 4 stars" is a rating, not a price)
    min_re = re.compile(
        rf"\b{MIN_WORDS}\s+(?P<c>{CUR_BEFORE})?(?P<n>{NUM})(?P<after>{CUR_AFTER})?{UNIT}", _FLAGS
    )
    for m in min_re.finditer(query):
        if _has_cur(m.group(0)) and claim(m) and res.price_min is None:
            res.price_min = _num(m.group("n"))

    # 4. rating: "4 stars and up", "at least 4.5 stars", "rated 4+"
    rate_re = re.compile(
        r"(?:at least\s+|rated\s+|minimum\s+)?(?P<r>[1-5](?:\.\d)?)\s*\+?\s*"
        r"(?:stars?\b(?:\s+(?:and up|or (?:more|higher|better)|\+))?|\+(?!\w))|"
        r"\brated\s+(?P<r2>[1-5](?:\.\d)?)\b",
        _FLAGS,
    )
    for m in rate_re.finditer(query):
        val = m.group("r") or m.group("r2")
        if val and claim(m) and res.min_rating is None:
            res.min_rating = float(val)

    rest = query
    for s, e in sorted(spans, reverse=True):
        rest = rest[:s] + " " + rest[e:]
    res.text = re.sub(r"\s+", " ", rest).strip(" ,.;-")
    return res
