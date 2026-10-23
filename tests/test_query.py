import json

import pytest

from cartrank.query import parser, regex_pass
from cartrank.query.schema import QueryIntent


@pytest.mark.parametrize(
    "q, pmin, pmax, rating",
    [
        ("stapler under $50", None, 50, None),
        ("pens less than 20 dollars", None, 20, None),
        ("notebook below 15", None, 15, None),
        ("desk organizer between 20 and 40", 20, 40, None),
        ("desk lamp $20-$40", 20, 40, None),
        ("monitor stand 25 to 60 dollars", 25, 60, None),
        ("printer over $200", 200, None, None),
        ("chair at least $100", 100, None, None),
        ("pens 4 stars and up", None, None, 4.0),
        ("at least 4.5 stars stapler", None, None, 4.5),
        ("stapler under $30 rated 4+", None, 30, 4.0),
        ("1,200 dollars max laptop desk", None, None, None),  # "max" must precede the number
        ("max $1,200 laptop desk", None, 1200, None),
    ],
)
def test_regex_extracts_price_and_rating(q, pmin, pmax, rating):
    r = regex_pass.parse(q)
    assert (r.price_min, r.price_max, r.min_rating) == (pmin, pmax, rating)


@pytest.mark.parametrize(
    "q",
    [
        "binder 3 inch",
        "under 3 inch binder",
        "paper 8.5 x 11",
        "pens 12 pack",
        "at least 4 stars is fine but no price here",  # rating only, no price
        "label tape 1/2 inch",
        "copy paper 500 sheets",
        "notebook between 3 and 5 inches",
    ],
)
def test_regex_does_not_invent_prices(q):
    r = regex_pass.parse(q)
    assert r.price_min is None and r.price_max is None


def test_regex_removes_matched_phrases_from_text():
    assert regex_pass.parse("black pens under $10").text == "black pens"
    assert regex_pass.parse("stapler between $10 and $20 4 stars and up").text == "stapler"


class FakeClient:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def complete(self, system, user):
        if self.exc:
            raise self.exc
        return json.dumps(self.payload), {"prompt_tokens": 100, "completion_tokens": 20}


def test_parser_regex_wins_over_llm_on_price():
    llm = {"route": "search", "text": "stapler", "price_max": 999, "brand": "Swingline"}
    res = parser.parse("swingline stapler under $30", FakeClient(llm))
    assert res.used_llm and res.intent.price_max == 30 and res.intent.brand == "Swingline"
    assert res.usage["prompt_tokens"] == 100


def test_parser_keeps_llm_price_when_regex_finds_none():
    llm = {"text": "stapler", "price_max": 25}
    assert parser.parse("cheap stapler", FakeClient(llm)).intent.price_max == 25


@pytest.mark.parametrize(
    "client",
    [None, FakeClient(exc=TimeoutError()), FakeClient(payload={"route": "bogus"}), FakeClient({})],
)
def test_parser_falls_back_to_regex_only(client):
    res = parser.parse("pens under $10", client)
    if client is not None and client.payload == {}:
        # an empty object is a valid QueryIntent: the LLM answered, regex still fills price
        assert res.used_llm and res.intent.price_max == 10
        return
    assert not res.used_llm and res.intent.route == "search"
    assert res.intent.text == "pens" and res.intent.price_max == 10


def test_intent_schema_defaults():
    q = QueryIntent(text="pens")
    assert q.route == "search" and q.must_have == [] and q.exclude == []


def test_parser_unwraps_single_item_list():
    class ListClient:
        def complete(self, system, user):
            return json.dumps([{"text": "pens", "exclude": ["clips"]}]), {}

    res = parser.parse("pens without clips", ListClient())
    assert res.used_llm and res.intent.exclude == ["clips"]


def test_fallback_chain_order_and_errors():
    from cartrank.query.llm_clients import FallbackClient, LLMError

    class Fake:
        def __init__(self, name, fail):
            self.name, self.fail = name, fail

        def complete(self, system, user):
            if self.fail:
                raise LLMError(f"{self.name}: HTTP 429")
            return "{}", {"provider": self.name}

    chain = FallbackClient([Fake("a", True), Fake("b", False), Fake("c", False)])
    text, usage = chain.complete("s", "u")
    assert usage["provider"] == "b" and usage["fallbacks_used"] == 1
    with pytest.raises(LLMError, match="a: HTTP 429; b: HTTP 429"):
        FallbackClient([Fake("a", True), Fake("b", True)]).complete("s", "u")
