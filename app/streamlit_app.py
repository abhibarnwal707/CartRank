"""CartRank: ask in a sentence, get ten ranked, see how. Search and recommendation over Amazon
Office Products listings."""

import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

# Streamlit Cloud secrets -> environment (a local .env is read by cartrank.config)
try:
    for _k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "CARTRANK_ARTIFACT_DIR"):
        if _k in st.secrets:
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:  # no secrets file locally
    pass

import ui  # noqa: E402

from cartrank.serve import examples as ex  # noqa: E402
from cartrank.serve.loader import load_pipeline  # noqa: E402

st.set_page_config(page_title="CartRank", layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{(HERE / 'style.css').read_text()}</style>")

EXAMPLES = ex.load()


@st.cache_resource(show_spinner="Loading the models. This happens once, on the first search.")
def get_pipeline():
    return load_pipeline(device="cpu")


@st.cache_data(max_entries=256, ttl=3600, show_spinner=False)
def cached_search(_pipe, query: str):
    return _pipe.search(query)


def pipeline_or_stop():
    try:
        return get_pipeline()
    except Exception as e:  # Streamlit Cloud redacts exception text, so explain on screen
        from cartrank.serve.artifacts import hub_help

        hub = type(e).__name__ in ("RepositoryNotFoundError", "HfHubHTTPError", "GatedRepoError")
        st.error(
            f"CartRank could not load its model files ({type(e).__name__}). "
            + (hub_help() if hub else "See the app logs for details.")
        )
        st.stop()


def show_example(i: int) -> None:
    st.session_state.current = (EXAMPLES[i], "example")


# ------------------------------------------------------------------------------------- state
if "current" not in st.session_state and EXAMPLES:
    st.session_state.current = (EXAMPLES[0], "example")
st.session_state.setdefault("liked", {})
llm_on = bool(
    (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    and not os.environ.get("CARTRANK_NO_LLM")
)

# ----------------------------------------------------------------------------- page, in order
st.html(ui.nav_html())
hero_slot = st.empty()

with st.container(key="intro"):
    lead_col, ex_col = st.columns([5, 6])
    lead_col.html(ui.lead_html())
    with ex_col:
        st.html(ui.examples_head_html())
        with st.container(key="examples"):
            for i, e in enumerate(EXAMPLES):
                st.button(
                    e.query.replace("$", "\\$"), key=f"ex{i}", on_click=show_example, args=(i,)
                )
        if not llm_on:
            st.html(
                '<p class="note">Language-model parsing is off here, so only price and rating '
                "phrases are understood.</p>"
            )
with st.container(key="ask"):
    with st.form("ask", border=False):
        c_in, c_go = st.columns([8, 2])
        query = c_in.text_input(
            "Search",
            key="q",
            placeholder="black gel pens without clips under $10",
            label_visibility="collapsed",
        )
        submitted = c_go.form_submit_button("Rank it", width="stretch")

if submitted and query.strip():
    pipe = pipeline_or_stop()
    st.session_state.current = (cached_search(pipe, query.strip()), "search")

current = st.session_state.get("current")
if current is None:
    st.html('<p class="note">No examples are bundled; type a query above.</p>')
    hero_slot.html(ui.hero_html([]))
else:
    resp, source = current
    hero_slot.html(ui.hero_html([r.image_url for r in resp.results]))
    st.html(ui.reading_html(resp, example=source == "example"))
    st.html(ui.results_html(resp))
    st.html(ui.hood_html(resp))

st.html(ui.how_html())

# ----------------------------------------------------------------------------------- picker
with st.container(key="taste"):
    left, right = st.columns(2)
    with left:
        st.html(ui.taste_statement_html())
        st.html('<p class="caption">Your picks</p>')
        for asin, title in list(st.session_state.liked.items()):
            if st.button(f"Remove  {title[:56]}".replace("$", "\\$"), key=f"rm_{asin}"):
                del st.session_state.liked[asin]
                st.rerun()
        if not st.session_state.liked:
            st.html('<p class="note">Nothing yet. Search on the right and add a few.</p>')
        with st.container(key="recommend"):
            recommend = st.button(
                "Recommend for me", disabled=not st.session_state.liked, width="stretch"
            )
    with right:
        st.html('<p class="caption taste-find">Find a product</p>')
        lookup = st.text_input(
            "Find a product",
            key="lookup",
            placeholder="pilot g2 pens",
            label_visibility="collapsed",
        )
        if lookup:
            pipe = pipeline_or_stop()
            for r in pipe.lookup(lookup, 5):
                label = f"Add  {r.title[:56]}  ·  \\${r.price:,.0f}"
                if st.button(label, key=f"add_{r.asin}"):
                    st.session_state.liked[r.asin] = r.title
                    st.rerun()
    if recommend:
        pipe = pipeline_or_stop()
        st.session_state.rec = pipe.recommend(list(st.session_state.liked))
    if "rec" in st.session_state:
        st.html(ui.picker_rows_html(st.session_state.rec.results))

st.html(ui.about_html())
st.html(ui.footer_html())
