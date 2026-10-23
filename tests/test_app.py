import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "models" / "bge-small-esci-ft"
SUBMIT = "FormSubmitter:ask-Rank it"


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CARTRANK_NO_LLM", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    return at


def page(at) -> str:
    return " ".join(h.value for h in at.get("html"))


def prices(at) -> list[float]:
    return [float(p) for p in re.findall(r'data-price="([\d.]+)"', page(at))]


def test_first_screen_needs_no_models_and_shows_an_example(app):
    html = page(app)
    assert "Ask in a sentence." in html and "ranked." in html
    assert "An example result, replayed" in html and html.count('class="row') == 10
    assert "Language-model parsing is off" in html  # keys are removed in this fixture


def test_example_buttons_swap_the_whole_result(app):
    labels = {b.key: b.label for b in app.button if b.key and b.key.startswith("ex")}
    key = next(k for k, v in labels.items() if "swingline stapler" in v and "4 stars" in v)
    app.button(key=key).click().run()
    assert not app.exception
    html = page(app)
    assert (
        "swingline stapler" in html and "4 stars and up" in html and html.count('class="row') == 10
    )


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="needs local model artifacts (not in CI)")
def test_search_applies_the_price_filter_to_every_row(app):
    app.text_input(key="q").set_value("stapler under $10")
    app.button(key=SUBMIT).click().run()
    assert not app.exception
    p = prices(app)
    assert p and all(x <= 10 for x in p)
    assert "You asked" in page(app)


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="needs local model artifacts (not in CI)")
def test_picker_adds_an_item_and_recommends(app):
    app.text_input(key="lookup").set_value("swingline stapler").run()
    add = [b for b in app.button if b.key and b.key.startswith("add_")]
    assert add
    add[0].click().run()
    assert app.session_state.liked
    [b for b in app.button if b.label == "Recommend for me"][0].click().run()
    assert not app.exception
    assert "because you liked" in page(app)
