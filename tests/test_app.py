from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "models" / "bge-small-esci-ft"
pytestmark = pytest.mark.skipif(
    not ARTIFACTS.exists(), reason="needs local model artifacts (not available in CI)"
)


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CARTRANK_NO_LLM", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    return at


def test_app_loads_in_regex_only_mode_with_warning(app):
    assert any("regex-only" in w.value.lower() for w in app.warning)


def test_search_shows_cards_with_price_filter(app):
    app.text_input(key="query").set_value("stapler under $10").run()
    assert not app.exception
    assert any("stapler" in c.value.lower() for c in app.caption if "Why this" in c.value)
    prices = [
        float(m.value.split("**$")[1].split("**")[0].replace(",", ""))
        for m in app.markdown
        if m.value.startswith("**$")
    ]
    assert prices and all(p <= 10 for p in prices)


def test_picker_adds_item_and_recommends(app):
    app.text_input(key="lookup").set_value("swingline stapler").run()
    add = [b for b in app.button if b.key and b.key.startswith("add_")]
    assert add
    add[0].click().run()
    assert app.session_state.liked
    [b for b in app.button if b.label == "Recommend for me"][0].click().run()
    assert not app.exception
    assert any("Recommended for you" in s.value for s in app.subheader)
