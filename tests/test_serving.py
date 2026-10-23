import numpy as np
import pytest
import torch

from cartrank.recsys import two_tower as tt
from cartrank.recsys.serving import TwoTowerServing, export_from_state_dict


@pytest.fixture
def served():
    n, d_text = 9, 6
    text = np.random.default_rng(0).random((n, d_text)).astype(np.float32)
    model = tt.TwoTower(n, text, dim=8, hidden=12, seed=1)
    arrays = export_from_state_dict(model.state_dict(), n)
    return model, TwoTowerServing(arrays, [f"A{i}" for i in range(n)])


def test_item_vectors_match_torch(served):
    model, s = served
    torch_vecs = model.all_item_vecs().numpy()
    assert np.allclose(s.vecs, torch_vecs, atol=1e-5)


def test_user_vector_matches_torch_and_ignores_unknown(served):
    model, s = served
    hist = torch.tensor([[1, 4, 7, 9, 9]])  # 9 = padding
    expected = model.user_vec(hist).detach().numpy()[0]
    assert np.allclose(s.user_vec(["A1", "A4", "A7", "ZZZ"]), expected, atol=1e-5)
    with pytest.raises(ValueError):
        s.user_vec(["nope"])


def test_recommend_and_similar_exclude_inputs_and_respect_allowed(served):
    _, s = served
    recs = s.recommend(["A1", "A2"], k=3)
    assert len(recs) == 3 and not {"A1", "A2"} & {a for a, _ in recs}
    only = s.recommend(["A1"], k=5, allowed={"A3", "A4"})
    assert {a for a, _ in only} == {"A3", "A4"}
    sim = s.similar("A5", k=4)
    assert "A5" not in {a for a, _ in sim} and sim[0][1] >= sim[-1][1]
