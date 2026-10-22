import pandas as pd

from cartrank.data.splits import leave_last_out, split_queries


def test_split_queries_no_leak_and_fraction():
    ids = pd.Series([i for i in range(100) for _ in range(3)])
    train, dev = split_queries(ids, dev_frac=0.1, seed=1)
    assert not train & dev
    assert len(dev) == 10
    assert train | dev == set(range(100))


def test_split_queries_deterministic():
    ids = pd.Series(range(50))
    assert split_queries(ids, seed=3) == split_queries(ids, seed=3)


def _interactions():
    rows = []
    for t, item in enumerate(["a", "b", "c", "d"]):
        rows.append(("u1", item, 100 - t))  # shuffled timestamps: order by time, not by row
    rows += [("u2", "x", 1), ("u2", "y", 2)]  # too short, dropped
    rows += [("u3", "p", 1), ("u3", "q", 2), ("u3", "r", 3)]
    return pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])


def test_leave_last_out_temporal_order():
    s = leave_last_out(_interactions())
    # u1 by time: d(97) < c(98) < b(99) < a(100)
    assert s["test"].query("user_id == 'u1'").parent_asin.tolist() == ["a"]
    assert s["dev"].query("user_id == 'u1'").parent_asin.tolist() == ["b"]
    assert sorted(s["train"].query("user_id == 'u1'").parent_asin) == ["c", "d"]
    dev_ts = s["dev"].set_index("user_id").timestamp
    test_ts = s["test"].set_index("user_id").timestamp
    for user, g in s["train"].groupby("user_id"):
        assert g.timestamp.max() < dev_ts[user] < test_ts[user]


def test_leave_last_out_drops_short_users_and_one_each():
    s = leave_last_out(_interactions())
    assert all("u2" not in df.user_id.values for df in s.values())
    assert s["test"].user_id.is_unique and s["dev"].user_id.is_unique
    assert set(s["test"].user_id) == set(s["dev"].user_id) == {"u1", "u3"}
