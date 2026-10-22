from cartrank.search.bm25 import Bm25Index


def test_bm25_ranks_matching_doc_first():
    idx = Bm25Index(
        ["p1", "p2", "p3"],
        ["red running shoes for men", "stainless steel water bottle", "blue yoga mat"],
    )
    assert idx.search("water bottle", k=2)[0] == "p2"
    assert idx.search("running shoe", k=1) == ["p1"]  # stemming: shoe ~ shoes


def test_bm25_empty_query_scores_zero():
    idx = Bm25Index(["p1"], ["red shoes"])
    assert idx.scores("the of").sum() == 0
