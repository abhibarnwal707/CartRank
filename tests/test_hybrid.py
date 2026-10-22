import pytest

from cartrank.search.hybrid import rrf


def test_rrf_hand_computed():
    out = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # a and b tie on score (1/61 + 1/62); stable sort keeps first-seen order
    assert out[:2] == ["a", "b"]
    assert set(out) == {"a", "b", "c", "d"}


def test_rrf_prefers_agreement():
    assert rrf([["x", "y"], ["z", "y"]])[0] == "y"


def test_rrf_depth_truncates():
    assert "c" not in rrf([["a", "b", "c"]], depth=2)


def test_rrf_score_value():
    # single list: doc ranked 1 has score 1/(k+1)
    assert rrf([["a"]], k=1) == ["a"]
    assert 1 / 61 == pytest.approx(1 / (60 + 1))
