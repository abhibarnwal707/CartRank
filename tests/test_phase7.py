import pytest

from cartrank.eval import run_phase7 as p7


def test_guard_blocks_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(p7, "LOG", tmp_path / "test_runs.md")
    monkeypatch.delenv("FORCE_TEST_RERUN", raising=False)
    p7.guard("stage-x")  # first time: fine
    p7.log_run("stage-x", "unit test")
    with pytest.raises(SystemExit, match="already run"):
        p7.guard("stage-x")
    p7.guard("stage-y")  # a different stage is unaffected
    monkeypatch.setenv("FORCE_TEST_RERUN", "1")
    p7.guard("stage-x")  # explicit crash-recovery override


def test_log_has_header_and_git_state(tmp_path, monkeypatch):
    monkeypatch.setattr(p7, "LOG", tmp_path / "test_runs.md")
    p7.log_run("s", "what")
    text = (tmp_path / "test_runs.md").read_text()
    assert text.startswith("# Test split runs") and "| s |" in text and "what" in text


def test_build_test_hist_appends_dev_item_last(monkeypatch):
    import pandas as pd

    from cartrank.recsys import data as recdata

    tr = pd.DataFrame(
        {
            "user_id": ["u"] * 3,
            "parent_asin": ["a", "b", "c"],
            "rating": 5.0,
            "timestamp": [1, 2, 3],
        }
    )
    dev = pd.DataFrame({"user_id": ["u"], "parent_asin": ["d"], "rating": 5.0, "timestamp": [4]})
    te = pd.DataFrame({"user_id": ["u"], "parent_asin": ["e"], "rating": 5.0, "timestamp": [5]})
    data = recdata.build(tr, dev, te)

    monkeypatch.setattr(pd, "read_parquet", lambda *_a, **_k: tr)
    hist = p7.build_test_hist(data, hist_len=5)
    items = list(data.item_ids)
    assert hist[0].tolist() == [
        data.n_items,
        items.index("a"),
        items.index("b"),
        items.index("c"),
        items.index("d"),
    ]
    assert items.index("e") not in hist[0]  # the test item never appears in the input
