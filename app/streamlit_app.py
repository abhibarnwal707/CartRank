"""CartRank demo: product search + recommendation over an Amazon Office Products catalog."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Streamlit Cloud secrets -> environment (a local .env is read by cartrank.config)
try:
    for _k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "CARTRANK_ARTIFACT_DIR"):
        if _k in st.secrets:
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:  # no secrets file locally
    pass

from cartrank.serve.loader import load_pipeline  # noqa: E402

st.set_page_config(page_title="CartRank", page_icon="🛒", layout="wide")

EXAMPLES = [
    "black gel pens without clips under $10",
    "swingline stapler 4 stars and up",
    "something like the swingline 747 stapler",
    "notebook not spiral bound between $5 and $15",
    "stappler",
]


@st.cache_resource(show_spinner="Loading models and indexes (first run only)...")
def get_pipeline():
    return load_pipeline(device="cpu")


@st.cache_data(max_entries=256, ttl=3600, show_spinner=False)
def cached_search(_pipe, query: str):
    return _pipe.search(query)


def card(r, col):
    with col:
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            if r.image_url:
                c1.image(r.image_url, width=96)
            c2.markdown(f"**{r.title[:110]}**")
            c2.markdown(
                f"**${r.price:,.2f}** · ★ {r.rating:.1f} ({r.rating_number:,} ratings)"
                + (f" · {r.store}" if r.store else "")
            )
            if r.why:
                c2.caption(f"Why this: {r.why}")


def show(resp, show_ref=True):
    for note in resp.notes:
        st.info(note)
    if show_ref and resp.reference is not None:
        st.markdown(f"Showing products similar to **{resp.reference.title[:100]}**")
    if not resp.results:
        st.warning("No products found.")
    cols = st.columns(2)
    for i, r in enumerate(resp.results):
        card(r, cols[i % 2])
    with st.expander("Debug: what the pipeline did"):
        st.markdown(
            f"**Route used:** `{resp.route_used}` · **LLM parser:** "
            f"{'on' if resp.llm_used else 'off (regex-only)'}"
        )
        st.json(resp.intent.model_dump())
        st.markdown("**Per-stage latency (ms)**")
        st.dataframe({k: [round(v, 1)] for k, v in resp.timings_ms.items()}, hide_index=True)
        st.markdown("**Candidates per stage**")
        st.json(resp.debug.get("candidates", {}))
        from cartrank.serve.sysinfo import info

        st.markdown("**Server**")
        st.json(
            {**info(), "startup_seconds (one-time, this container)": getattr(pipe, "startup", None)}
        )
        parser_info = resp.debug.get("parser", {})
        if parser_info.get("usage"):
            u = parser_info["usage"]
            st.caption(
                f"LLM: {u.get('provider')} / {u.get('model')} · "
                f"{u.get('prompt_tokens')} prompt + {u.get('completion_tokens')} completion tokens"
            )
        elif parser_info.get("error"):
            st.caption(f"LLM not used ({parser_info['error']})")


try:
    pipe = get_pipeline()
except Exception as e:  # Streamlit Cloud redacts exception text, so explain on screen
    from cartrank.serve.artifacts import hub_help

    hub_error = type(e).__name__ in ("RepositoryNotFoundError", "HfHubHTTPError", "GatedRepoError")
    st.error(
        "CartRank could not load its model files "
        f"({type(e).__name__}). " + (hub_help() if hub_error else "See the app logs for details.")
    )
    st.stop()
st.title("🛒 CartRank")
st.caption(
    "Hybrid BM25 + fine-tuned dense retrieval, cross-encoder reranking, a two-tower recommender "
    "and an LLM query parser, over ~52K Amazon Office Products items (Amazon Reviews 2023)."
)
if pipe.client is None:
    st.warning(
        "Running in regex-only mode: no LLM API keys are configured, so only price and "
        "rating phrases are parsed."
    )

# ----------------------------------------------------------------------------- sidebar picker
if "liked" not in st.session_state:
    st.session_state.liked = {}
with st.sidebar:
    st.header("Pick items you like")
    st.caption("Search, add a few items, then get recommendations from the two-tower model.")
    lookup = st.text_input("Find an item", key="lookup", placeholder="e.g. pilot g2 pens")
    if lookup:
        for r in pipe.lookup(lookup, 5):
            if st.button(f"+ {r.title[:48]} (${r.price:.0f})", key=f"add_{r.asin}"):
                st.session_state.liked[r.asin] = r.title
    for asin, title in list(st.session_state.liked.items()):
        if st.button(f"✕ {title[:50]}", key=f"rm_{asin}"):
            del st.session_state.liked[asin]
            st.rerun()
    if st.button("Recommend for me", type="primary", disabled=not st.session_state.liked):
        st.session_state.rec = pipe.recommend(list(st.session_state.liked))

# ----------------------------------------------------------------------------- benchmark
if st.query_params.get("bench") == "1":
    import json

    from cartrank.serve.bench import run_benchmark
    from cartrank.serve.pipeline import Pipeline

    with st.sidebar.expander("Maintainer: benchmark (LLM off, no API tokens)"):
        if st.button("Run 50-query benchmark"):
            qfile = Path(__file__).resolve().parents[1] / "evals" / "queries" / "demo_queries.jsonl"
            queries = [json.loads(line)["query"] for line in qfile.read_text().splitlines()]
            offline = Pipeline(pipe.cs, pipe.tt, None, ce_batch=pipe.ce_batch)
            with st.spinner("Running 50 queries..."):
                result = run_benchmark(offline, queries)
            result["startup_seconds"] = getattr(pipe, "startup", None)
            st.session_state.bench_result = result
        if "bench_result" in st.session_state:
            st.code(json.dumps(st.session_state.bench_result, indent=1), language="json")

# ----------------------------------------------------------------------------- main
query = st.text_input(
    "Search", key="query", placeholder="Try: black gel pens without clips under $10"
)
ex_cols = st.columns(len(EXAMPLES))
for c, ex in zip(ex_cols, EXAMPLES, strict=True):
    if c.button(ex, key=f"ex_{ex}"):
        query = ex

if query:
    show(cached_search(pipe, query))
elif "rec" in st.session_state:
    st.subheader("Recommended for you")
    show(st.session_state.rec, show_ref=False)
else:
    st.markdown("Type a query above, click an example, or pick items you like in the sidebar.")
