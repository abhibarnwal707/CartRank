import pytest

from cartrank.eval.parser_eval import field_match, score, token_f1
from cartrank.query.schema import QueryIntent


def test_token_f1():
    assert token_f1("black gel pens", "black gel pens") == 1.0
    assert token_f1("black gel pens", "gel pens") == pytest.approx(0.8)
    assert token_f1("pens", "stapler") == 0.0


def test_field_match_normalises_case_and_sets():
    g = QueryIntent(brand="Swingline", exclude=["clips", "Tabs"])
    p = QueryIntent(brand=" swingline ", exclude=["tabs", "clips"])
    assert field_match("brand", g, p) and field_match("exclude", g, p)
    assert not field_match("price_max", g, QueryIntent(price_max=5))


def test_score_overall_and_group():
    g = QueryIntent(text="pens", price_max=10)
    rows = [
        {"group": "price", "gold": g, "pred": QueryIntent(text="pens", price_max=10)},
        {"group": "price", "gold": g, "pred": QueryIntent(text="pens")},
    ]
    s = score(rows)
    assert s["overall"]["price_max"] == 0.5 and s["overall"]["route"] == 1.0
    assert s["overall"]["constraint_intent_exact"] == 0.5
    assert s["by_group"]["price"]["n"] == 2
