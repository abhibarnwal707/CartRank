from cartrank.eval.constraints_eval import OBJECTIVE, check, mentions, n_constraints, satisfies_all
from cartrank.query.schema import QueryIntent

ITEM = {
    "title": "Swingline Desk Stapler, no staples included",
    "features": ["Cordless, quiet"],
    "price": 18.5,
    "average_rating": 4.4,
    "store": "Swingline",
}


def test_mentions_handles_negation_and_multiword():
    assert mentions("pens with clips", "clips")
    assert not mentions("pens without clips", "clips")
    assert not mentions("tabs free folders", "tabs")
    assert mentions("self-adhesive envelopes", "self adhesive")
    assert not mentions("", "x")


def test_check_each_constraint():
    c = QueryIntent(
        price_max=20, min_rating=4, brand="Swingline", must_have=["cordless"], exclude=["staples"]
    )
    r = check(ITEM, c)
    assert r == {"price": True, "rating": True, "brand": True, "must_have": True, "exclude": True}
    assert check(ITEM, QueryIntent(price_max=10))["price"] is False
    assert check(ITEM, QueryIntent(min_rating=4.5))["rating"] is False
    assert check(ITEM, QueryIntent())["price"] is None


def test_satisfies_all_and_kind_filter():
    c = QueryIntent(price_max=10, brand="swingline", exclude=["quiet"])
    assert not satisfies_all(ITEM, c)
    assert satisfies_all(ITEM, c, kinds=("brand",))
    assert not satisfies_all(ITEM, c, kinds=OBJECTIVE)  # price fails


def test_n_constraints_counts_terms():
    c = QueryIntent(price_max=5, brand="x", must_have=["a", "b"], exclude=["c"])
    assert n_constraints(c) == 5 and n_constraints(c, OBJECTIVE) == 2
