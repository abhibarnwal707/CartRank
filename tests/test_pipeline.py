import json

import numpy as np
import pytest

from cartrank.query.schema import QueryIntent
from cartrank.recsys import two_tower as tt_mod
from cartrank.recsys.serving import TwoTowerServing, export_from_state_dict
from cartrank.serve.pipeline import Pipeline, constraint_notes, matched_terms, why_search

ITEMS = {
    "A0": ("Swingline Desk Stapler", "Swingline", 18.0, 4.5, "metal stapler"),
    "A1": ("Cheap Plastic Stapler", "Acme", 6.0, 3.9, "plastic stapler small"),
    "A2": ("Gel Pens Black, 12 pack, without clips", "Pilot", 9.0, 4.6, "smooth gel ink pens"),
    "A3": ("Gel Pens Blue with clips", "Bic", 7.0, 4.2, "gel ink pens clips"),
    "A4": ("Desk Organizer Mesh", "Acme", 25.0, 4.0, "mesh desk organizer"),
}


class FakeBM25:
    def __init__(self, cs):
        self.cs = cs

    def search(self, q, n):
        words = set(q.lower().split())
        score = {a: len(words & set(self.cs.text[a].lower().split())) for a in self.cs.ids}
        return sorted(self.cs.ids, key=lambda a: -score[a])[:n]


class FakeCE:
    def predict(self, pairs, batch_size=64):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


class FakeCS:
    ids = list(ITEMS)
    depth, k, ce = 5, 10, None

    def __init__(self):
        self.text = {a: f"{v[0]}. {v[1]}. {v[4]}" for a, v in ITEMS.items()}
        self.bm25 = FakeBM25(self)

    def dense(self, q, n):
        return self.bm25.search(q, n)[::-1][:n] if False else self.bm25.search(q, n)

    def item(self, a):
        t, s, p, r, f = ITEMS[a]
        return {"title": t, "store": s, "price": p, "average_rating": r, "features": [f]}

    def meta(self, a):
        return {"rating_number": 10, "image_url": None}


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        return json.dumps(self.payload), {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def pipe():
    n = 5
    model = tt_mod.TwoTower(n, np.random.default_rng(0).random((n, 4)).astype(np.float32), dim=8)
    served = TwoTowerServing(export_from_state_dict(model.state_dict(), n), list(ITEMS))
    cs = FakeCS()
    cs.ce = FakeCE()
    return Pipeline(cs, served, FakeClient({"text": "gel pens", "price_max": 8}), pool=10)


def test_matched_terms_and_notes():
    item = FakeCS().item("A2")
    assert matched_terms("black gel pens for kids", item) == ["black", "gel", "pens"]
    c = QueryIntent(price_max=10, exclude=["clips"], brand="Pilot")
    assert constraint_notes(item, c)[0].startswith("price $9.00")
    assert "semantic match" in why_search(item, "zzz", QueryIntent(), False)


def test_search_applies_hard_filters_and_templated_why(pipe):
    resp = pipe.search("gel pens under $8")
    assert resp.route_used == "search" and resp.llm_used
    asins = [r.asin for r in resp.results]
    assert asins[0] == "A3"  # best text match among the products passing the price filter
    assert set(asins) == {"A1", "A3"}  # A0, A2, A4 cost more than $8 and are filtered out
    assert "price $7.00" in resp.results[0].why and "$8" in resp.results[0].why
    assert all(r.price <= 8 for r in resp.results)
    assert resp.debug["candidates"]["after_filter"] == 2
    assert set(resp.timings_ms) >= {"parse", "bm25", "dense", "fuse", "filter", "rerank", "total"}


def test_search_relaxes_when_nothing_matches(pipe):
    pipe.client = FakeClient({"text": "stapler", "price_max": 1})
    resp = pipe.search("stapler under $1")
    assert resp.results and any("relaxed" in n for n in resp.notes)
    assert "price $" not in resp.results[0].why  # constraints not claimed when relaxed


def test_regex_only_mode_note(pipe):
    pipe.client = None
    resp = pipe.search("desk organizer under $30")
    assert not resp.llm_used and any("Regex-only" in n for n in resp.notes)
    assert resp.intent.price_max == 30


def test_similar_to_and_recommend_routes(pipe):
    pipe.client = FakeClient({"route": "similar_to", "text": "swingline desk stapler"})
    resp = pipe.search("something like the swingline stapler")
    assert resp.route_used == "similar_to" and resp.reference.asin == "A0"
    assert "A0" not in [r.asin for r in resp.results] and "similar to" in resp.results[0].why
    rec = pipe.recommend(["A0", "A1"])
    assert rec.route_used == "recommend" and not {"A0", "A1"} & {r.asin for r in rec.results}
    assert rec.results[0].why.startswith("because you liked")


def test_recommend_route_without_history_falls_back_to_search(pipe):
    pipe.client = FakeClient({"route": "recommend", "text": "desk organizer"})
    resp = pipe.search("recommend something for my desk")
    assert resp.route_used == "search" and any("sidebar" in n for n in resp.notes)
