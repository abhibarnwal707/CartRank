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
CATALOG_IMAGES = [r.image_url for e in EXAMPLES for r in e.results]

st.html(ui.bar_html())

# Hero: the catalog as a wall, the search docked on it. The wall is drawn after the form runs so a
# new search lights up its own ten.
with st.container(key="hero"):
    wall_slot = st.empty()
    with st.container(key="dock"):
        st.html(ui.title_html())
        with st.form("ask", border=False):
            c_in, c_go = st.columns([8, 2])
            query = c_in.text_input(
                "Search",
                key="q",
                placeholder="black gel pens without clips under $10",
                label_visibility="collapsed",
            )
            submitted = c_go.form_submit_button("Rank it", width="stretch")
        with st.container(key="examples", horizontal=True):
            for i, e in enumerate(EXAMPLES):
                st.button(
                    e.query.replace("$", "\\$"), key=f"ex{i}", on_click=show_example, args=(i,)
                )
        if not llm_on:
            st.html(
                '<p class="dock-note">Language model off: only prices and ratings are read.</p>'
            )

if submitted and query.strip():
    pipe = pipeline_or_stop()
    st.session_state.current = (cached_search(pipe, query.strip()), "search")

current = st.session_state.get("current")
resp, source = current if current else (None, "")
wall_slot.html(ui.wall_html([r.image_url for r in resp.results] if resp else [], CATALOG_IMAGES))

# Results: index with preview, then the two explanations behind buttons.
with st.container(key="results"):
    if resp is None:
        st.html('<p class="note">No examples are bundled; type a query above.</p>')
    else:
        st.html(ui.results_html(resp, example=source == "example"))
        with st.container(key="tools", horizontal=True):
            with st.popover("How it narrowed", key="pop_funnel"):
                st.html(ui.funnel_html(resp))
            with st.popover("What it understood", key="pop_hood"):
                st.html(ui.hood_html(resp))

# Recommend: say what you like, it finds more.
with st.container(key="picker"):
    left, right = st.columns([5, 7])
    left.html(ui.picker_intro_html())
    with right, st.container(key="picker_work"):
        lookup = st.text_input(
            "Find a product",
            key="lookup",
            placeholder="Find a product: pilot g2 pens",
            label_visibility="collapsed",
        )
        if lookup:
            found = pipeline_or_stop().lookup(lookup, 5)
            for col, r in zip(st.columns(5), found, strict=False):
                with col:
                    st.html(ui.mini_html(r.title, r.image_url, r.price))
                    if st.button(
                        "Add",
                        key=f"add_{r.asin}",
                        width="stretch",
                        disabled=len(st.session_state.liked) >= 3,
                    ):
                        st.session_state.liked[r.asin] = (r.title, r.image_url)
                        st.rerun()
        picks = list(st.session_state.liked.items())
        with st.container(key="slots"):
            for n, col in enumerate(st.columns(3), 1):
                with col:
                    if n <= len(picks):
                        asin, (title, image) = picks[n - 1]
                        st.html(ui.mini_html(title, image))
                        if st.button("Remove", key=f"rm_{asin}", width="stretch"):
                            del st.session_state.liked[asin]
                            st.rerun()
                    else:
                        st.html(ui.slot_html(n))
        with st.container(key="recommend"):
            recommend = st.button(
                "Recommend for me", disabled=not st.session_state.liked, width="stretch"
            )
    if recommend:
        st.session_state.rec = pipeline_or_stop().recommend(list(st.session_state.liked))
    if "rec" in st.session_state:
        st.html(ui.recs_html(st.session_state.rec.results))

# How it ranks: four fields, one step each; the detail sits behind a button.
with st.container(key="how"):
    for i, col in enumerate(st.columns(4)):
        with col, st.container(key=f"field{i}"):
            st.html(ui.field_html(i, resp))
            with st.popover("More", key=f"pop_how{i}"):
                st.html(f'<p class="pop-text">{ui.esc(ui.how_more(i))}</p>')

# The numbers: three measured figures on stepped tiles, each with how it was measured.
with st.container(key="numbers"):
    for i, col in enumerate(st.columns([6, 3, 3])):
        with col, st.container(key=f"stat{i}"):
            if i == 1:
                st.html(ui.numbers_head_html())
            st.html(ui.stat_html(i))
            with st.popover("How measured", key=f"pop_stat{i}"):
                st.html(f'<p class="pop-text">{ui.esc(ui.stat_more(i))}</p>')

st.html(ui.footer_html())
