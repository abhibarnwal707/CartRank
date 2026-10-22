import math

import pytest

from cartrank.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

LABELS = {"a": "E", "b": "S", "c": "C", "d": "I", "e": "E"}


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "e", "b", "c", "d"], LABELS) == pytest.approx(1.0)
    assert ndcg_at_k(["d", "x", "y"], LABELS) == 0.0


def test_ndcg_hand_computed():
    # ranking [b, a]: dcg = 0.1/log2(2) + 1/log2(3); ideal [a, e, b, c]: 1 + 1/log2(3) + 0.1/2 + ...
    ranked = ["b", "a"]
    dcg = 0.1 + 1 / math.log2(3)
    ideal = 1 + 1 / math.log2(3) + 0.1 / math.log2(4) + 0.01 / math.log2(5)
    assert ndcg_at_k(ranked, LABELS) == pytest.approx(dcg / ideal)


def test_ndcg_unjudged_is_zero_gain_and_k_cutoff():
    assert ndcg_at_k(["x", "a"], LABELS, k=1) == 0.0
    assert ndcg_at_k(["x", "a"], LABELS, k=2) > 0.0


def test_ndcg_no_relevant_labels():
    assert ndcg_at_k(["a"], {"a": "I"}) == 0.0


def test_recall():
    assert recall_at_k(["a", "x"], LABELS, k=2) == 0.5
    assert recall_at_k(["a", "e"], LABELS, k=1) == 0.5
    assert recall_at_k(["a"], {"a": "S"}) is None


def test_mrr():
    assert mrr_at_k(["b", "x", "a"], LABELS) == pytest.approx(1 / 3)
    assert mrr_at_k(["b", "c"], LABELS) == 0.0
    assert mrr_at_k(["x"] * 10 + ["a"], LABELS, k=10) == 0.0


def test_evaluate_skips_queries_without_exact_for_recall():
    qrels = {1: LABELS, 2: {"z": "S"}}
    out = evaluate({1: ["a", "e"], 2: ["z"]}, qrels)
    assert out["n_queries"] == 2
    assert out["recall@100"] == 1.0
    assert out["mrr@10"] == pytest.approx(0.5)
