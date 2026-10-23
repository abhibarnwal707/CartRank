"""Query parser: regex pre-pass + one LLM call with structured output, with a regex-only fallback.

The LLM client is pluggable (`LLMClient`), so the app never breaks without the API: on any
failure or timeout the parser returns the regex result with route "search".
"""

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from cartrank.query import regex_pass
from cartrank.query.llm_clients import LLMError
from cartrank.query.schema import QueryIntent

PROMPT_V1 = """You convert a shopping query into JSON with exactly these keys:
route: "search" (default), "recommend" (user wants suggestions with no product named, e.g.
"what should I buy next", "recommend something for my desk") or "similar_to" (user names a
reference product and wants alternatives, e.g. "something like the Swingline 747").
text: the product phrase to search for, in English (translate other languages; fix obvious
typos). For similar_to this is the reference product. Remove price and rating phrases.
price_min, price_max: numbers in US dollars or null. brand: string or null.
category: a short product category or null. min_rating: 1-5 or null.
must_have: list of required product attributes (e.g. "refillable"). exclude: list of things the
user does NOT want (from "without X", "no X", "not Y"). Output JSON only."""

PROMPT_V2 = """You convert a shopping query into JSON with exactly these keys:
route, text, price_min, price_max, brand, category, min_rating, must_have, exclude.

route: "search" for ANY query that names or describes a product to find, even with price,
star-rating or "without" filters (this is the default). "recommend" ONLY when no product type is
named and the user asks for suggestions or ideas. "similar_to" ONLY when the user asks for
alternatives to, or something like, a named product.
text: the product phrase to search for, in English (translate other languages and fix typos). Keep
the brand and product attributes in it; remove price, rating and exclusion phrases.
price_min, price_max: US dollars or null. Never treat other numbers (sizes, counts) as prices.
brand: the brand name if the query mentions one, even when it is part of the product phrase
("sharpie markers" -> "Sharpie"); fix brand typos; otherwise null.
category: short product category or null. min_rating: 1-5 or null (only from an explicit rating).
must_have: required features given as adjectives or "with X" (cordless, waterproof, refillable,
"with pockets"). Do not invent features that are not in the query.
exclude: things the user does not want, from "without/no/not/except": the unwanted thing only,
e.g. "pens without clips" -> ["clips"].

Examples (output JSON only):
"red staplers under $15 without staples" -> {"route":"search","text":"red staplers","price_min":null,"price_max":15,"brand":null,"category":null,"min_rating":null,"must_have":[],"exclude":["staples"]}
"cordless bostitch stapler 4+ stars" -> {"route":"search","text":"cordless bostitch stapler","price_min":null,"price_max":null,"brand":"Bostitch","category":null,"min_rating":4,"must_have":["cordless"],"exclude":[]}
"alternatives to the pilot frixion pen" -> {"route":"similar_to","text":"pilot frixion pen","price_min":null,"price_max":null,"brand":"Pilot","category":null,"min_rating":null,"must_have":[],"exclude":[]}
"suggest gifts for a coworker" -> {"route":"recommend","text":"gifts for a coworker","price_min":null,"price_max":null,"brand":null,"category":null,"min_rating":null,"must_have":[],"exclude":[]}"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
SYSTEM_PROMPT = PROMPT_V2  # current


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> tuple[str, dict]:
        """Return (json_text, usage) where usage has prompt_tokens and completion_tokens."""


@dataclass
class ParseResult:
    intent: QueryIntent
    used_llm: bool
    latency_s: float
    usage: dict = field(default_factory=dict)
    error: str | None = None


def _merge(intent: QueryIntent, rx: regex_pass.RegexResult) -> QueryIntent:
    """The deterministic pre-pass wins on price and rating."""
    update = {}
    if rx.price_min is not None or rx.price_max is not None:
        update["price_min"], update["price_max"] = rx.price_min, rx.price_max
    if rx.min_rating is not None:
        update["min_rating"] = rx.min_rating
    return intent.model_copy(update=update)


def parse(query: str, client: LLMClient | None, system: str | None = None) -> ParseResult:
    t0 = time.perf_counter()
    rx = regex_pass.parse(query)
    fallback = QueryIntent(
        route="search",
        text=rx.text or query,
        price_min=rx.price_min,
        price_max=rx.price_max,
        min_rating=rx.min_rating,
    )
    if client is None:
        return ParseResult(fallback, False, time.perf_counter() - t0, error="no_client")
    try:
        raw, usage = client.complete(system or SYSTEM_PROMPT, query)
        obj = json.loads(raw)
        if isinstance(obj, list) and len(obj) == 1:  # some models wrap the object in a list
            obj = obj[0]
        intent = QueryIntent.model_validate(obj)
    except (LLMError, ValidationError, ValueError, TimeoutError, OSError, KeyError) as e:
        return ParseResult(fallback, False, time.perf_counter() - t0, error=type(e).__name__)
    return ParseResult(_merge(intent, rx), True, time.perf_counter() - t0, usage=usage)
