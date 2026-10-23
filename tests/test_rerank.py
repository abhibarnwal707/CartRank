from cartrank.search.rerank import rerank


def test_rerank_orders_head_only():
    scores = {(1, "a"): 0.1, (1, "b"): 0.9, (1, "c"): 0.5, (1, "d"): 9.0}
    assert rerank(1, ["a", "b", "c", "d"], scores, k=3) == ["b", "c", "a", "d"]


def test_rerank_k_larger_than_list():
    scores = {(1, "a"): 1.0, (1, "b"): 2.0}
    assert rerank(1, ["a", "b"], scores, k=50) == ["b", "a"]
