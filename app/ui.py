"""HTML builders for the CartRank page. Pure functions: data in, markup out.

Every string that comes from the catalog, the model or the visitor goes through `esc`, because the
markup is rendered with unsafe HTML enabled.
"""

from __future__ import annotations

import html
import math

REPO = "https://github.com/abhibarnwal707/CartRank"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def money(x: float) -> str:
    return f"${x:,.2f}" if x != int(x) else f"${int(x):,}"


# --------------------------------------------------------------------------- navigation + hero
def nav_html() -> str:
    return (
        '<header class="nav" id="top">'
        '<a class="wordmark" href="#top">CartRank</a>'
        '<nav aria-label="Sections">'
        '<a href="#how">How it ranks</a><a href="#taste">Recommend</a>'
        f'<a href="{REPO}" target="_blank" rel="noopener">Code</a></nav></header>'
    )


def strip_html(images: list[str | None]) -> str:
    """The ten result photos as a one-line-high strip that sits in the headline where 'ten' goes."""
    tiles = "".join(
        f'<span class="tile"><img src="{esc(u)}" alt="" loading="eager"></span>'
        for u in (images or [])[:10]
        if u
    )
    return f'<span class="strip" role="img" aria-label="ten">{tiles}</span>'


def hero_html(images: list[str | None]) -> str:
    return (
        '<section class="hero">'
        '<h1 class="display">'
        '<span class="line"><span data-line style="--i:0">Ask in a sentence.</span></span>'
        '<span class="line"><span data-line style="--i:1">Get '
        f"{strip_html(images)}"
        '<span class="sr">ten</span> ranked.</span></span></h1></section>'
    )


def lead_html() -> str:
    return (
        '<p class="lead" data-load style="--d:420ms">CartRank reads what you type, narrows 51,814 '
        "office products to ten, and tells you why each one made the list.</p>"
    )


def examples_head_html() -> str:
    return '<h2 class="lead">Or start from one of these.</h2>'


# --------------------------------------------------------------------------------- reading
def tokens_for(intent) -> list[tuple[str, str]]:
    """(kind, label) for every constraint the parser found; kind is 'need' or 'skip'."""
    out: list[tuple[str, str]] = []
    if intent.route == "similar_to":
        out.append(("skip", "similar to"))
    lo, hi = intent.price_min, intent.price_max
    if lo is not None and hi is not None:
        out.append(("need", f"{money(lo)}–{money(hi)}"))
    elif hi is not None:
        out.append(("need", f"up to {money(hi)}"))
    elif lo is not None:
        out.append(("need", f"from {money(lo)}"))
    if intent.min_rating is not None:
        out.append(("need", f"{intent.min_rating:g} stars and up"))
    for m in intent.must_have:
        out.append(("need", m))
    for e in intent.exclude:
        out.append(("skip", f"no {e}"))
    return out


def reading_html(resp, example: bool) -> str:
    """The query as a sentence in heading type, its constraints turned into flat tokens, with the
    funnel of how the list was narrowed on the other side."""
    intent = resp.intent
    text = intent.text or resp.query
    toks = "".join(f'<span class="tok {k}">{esc(t)}</span> ' for k, t in tokens_for(intent))
    label = (
        "An example result, replayed. Type your own above."
        if example
        else f"You asked: “{esc(resp.query)}”"
    )
    notes = "".join(f'<p class="note">{esc(n)}</p>' for n in resp.notes)
    return (
        '<section class="reading band">'
        '<div class="reading-left">'
        f'<h2 class="statement">{esc(text)} {toks}</h2>'
        f'<p class="caption">{label}</p>{notes}</div>'
        f'<div class="reading-right">{funnel_html(resp)}</div></section>'
    )


# ---------------------------------------------------------------------------------- funnel
def funnel_stages(resp) -> list[tuple[str, int, str]]:
    """(label, count, kind) from the counts the pipeline recorded for this very query."""
    c = resp.debug.get("candidates", {})
    cat = c.get("catalog_size", 51814)
    if resp.route_used == "similar_to":
        return [
            (f"{cat:,} products in the catalog", cat, "base"),
            (
                f"{c.get('neighbours_pool', 0):,} nearest to the reference item",
                c.get("neighbours_pool", 0),
                "mid",
            ),
            (f"{c.get('after_filter', 0):,} pass your filters", c.get("after_filter", 0), "mid"),
            (f"{len(resp.results)} shown", len(resp.results), "end"),
        ]
    return [
        (f"{cat:,} products in the catalog", cat, "base"),
        (
            f"{c.get('fused', 0):,} candidates: keywords and meaning, merged",
            c.get("fused", 0),
            "mid",
        ),
        (f"{c.get('after_filter', 0):,} pass your filters", c.get("after_filter", 0), "mid"),
        (f"{c.get('reranked', 0)} re-read by the cross-encoder", c.get("reranked", 0), "mid"),
        (f"{len(resp.results)} shown", len(resp.results), "end"),
    ]


def funnel_html(resp) -> str:
    stages = [s for s in funnel_stages(resp) if s[1] > 0]
    if not stages:
        return ""
    top = math.log10(max(n for _, n, _ in stages) + 1)
    rows = []
    for label, n, kind in stages:
        w = max(0.04, math.log10(n + 1) / top)
        rows.append(
            f'<li class="stage {kind}"><span class="stage-label">{esc(label)}</span>'
            f'<span class="stage-bar"><span style="--w:{w:.3f}"></span></span></li>'
        )
    return (
        '<div class="funnel" aria-label="How the list was narrowed">'
        f"<ol>{''.join(rows)}</ol>"
        '<p class="caption">Bar length is the log of the count. Counts are from this query.</p>'
        "</div>"
    )


# ---------------------------------------------------------------------------------- results
def _relative(scores: list[float | None]) -> list[float]:
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return [1.0 for _ in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [0.1 + 0.9 * ((s - lo) / span) if s is not None else 0.1 for s in scores]


def row_html(rank: int, r, rel: float, first: bool) -> str:
    img = f'<img src="{esc(r.image_url)}" alt="" loading="lazy">' if r.image_url else ""
    store = f'<p class="store">{esc(r.store)}</p>' if r.store else ""
    return (
        f'<article class="row{" first" if first else ""}" style="--i:{rank}" '
        f'data-price="{r.price:.2f}" data-rank="{rank}">'
        f'<div class="rank" aria-label="Rank {rank}">{rank}</div>'
        f'<div class="photo">{img}</div>'
        '<div class="info">'
        f'<p class="title">{esc(r.title[:120])}</p>'
        f'<p class="meta">{r.rating:.1f} of 5 · {r.rating_number:,} ratings</p>'
        f'<p class="why">{esc(r.why)}</p>'
        f'<div class="score" role="img" aria-label="Score relative to the others in this list">'
        f'<span style="--w:{rel:.3f}"></span></div></div>'
        f'<div class="price"><p class="figure">{money(r.price)}</p>{store}</div>'
        "</article>"
    )


def results_html(resp) -> str:
    if not resp.results:
        return '<section class="results band"><p class="note">No products found.</p></section>'
    rel = _relative([r.score for r in resp.results])
    rows = "".join(row_html(i, r, rel[i - 1], i == 1) for i, r in enumerate(resp.results, 1))
    return (
        f'<section class="results band" aria-label="Ranked results">{rows}'
        '<p class="caption">Bars compare each result with the others in this list; they are '
        "not percentages.</p></section>"
    )


def picker_rows_html(results) -> str:
    rel = _relative([r.score for r in results])
    return "".join(row_html(i, r, rel[i - 1], i == 1) for i, r in enumerate(results, 1))


def hood_html(resp) -> str:
    """The parse, the timings and the model that parsed it, as plain text in a disclosure."""
    i = resp.intent
    parsed = [("route", resp.route_used), ("search text", i.text)]
    if i.price_min is not None:
        parsed.append(("price from", money(i.price_min)))
    if i.price_max is not None:
        parsed.append(("price up to", money(i.price_max)))
    if i.min_rating is not None:
        parsed.append(("minimum rating", f"{i.min_rating:g} stars"))
    for k in ("brand", "category"):
        if getattr(i, k):
            parsed.append((k, getattr(i, k)))
    if i.must_have:
        parsed.append(("must have", ", ".join(i.must_have)))
    if i.exclude:
        parsed.append(("exclude", ", ".join(i.exclude)))
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in parsed)
    if resp.timings_ms:
        t = ", ".join(f"{k} {v:.0f} ms" for k, v in resp.timings_ms.items() if k != "total")
        rows += f"<dt>stages</dt><dd>{esc(t)}</dd>"
    u = resp.debug.get("parser", {}).get("usage")
    if u:
        rows += (
            f"<dt>parsed by</dt><dd>{esc(u.get('provider'))} / {esc(u.get('model'))}, "
            f"{esc(u.get('prompt_tokens'))} + {esc(u.get('completion_tokens'))} tokens</dd>"
        )
    return f'<details class="hood"><summary>Under the hood</summary><dl>{rows}</dl></details>'


# ---------------------------------------------------------------------------------- how it works
def _squares(kinds: str) -> str:
    return "".join(f'<i class="sq {k}"></i>' for k in kinds)


def how_html() -> str:
    panels = [
        (
            "field-surface",
            "Read",
            "A language model turns your sentence into filters: a price range, a rating, a brand, "
            "and what to leave out.",
            '<div class="art art-read"><span class="w">black gel pens</span>'
            '<span class="tok skip">no clips</span><span class="tok need">up to $10</span></div>',
        ),
        (
            "field-accent",
            "Retrieve",
            "Keyword search and a model fine-tuned on Amazon's shopping labels each pull their best "
            "200. The two lists are merged into one.",
            f'<div class="art art-merge"><div>{_squares("aabab")}</div><div>{_squares("bbaab")}</div>'
            f'<div class="out">{_squares("abababab")}</div></div>',
        ),
        (
            "field-ink",
            "Filter",
            "Hard filters drop whatever breaks your rules, so “under $10” never returns $12.",
            f'<div class="art art-filter"><div>{_squares("kxkkxkxk")}</div>'
            f"<div>{_squares('kkxkkkxk')}</div><div>{_squares('xkkxkkkk')}</div></div>",
        ),
        (
            "field-surface",
            "Rerank",
            "A cross-encoder reads your query next to each of the best 50 and puts them in order.",
            '<div class="art art-rank">'
            + "".join(f'<span style="--w:{w}"></span>' for w in (1.0, 0.82, 0.66, 0.5, 0.36))
            + "</div>",
        ),
    ]
    out = []
    for cls, verb, sentence, art in panels:
        out.append(
            f'<section class="panel {cls}"><p class="verb" aria-hidden="true">{verb}</p>'
            f'<h2 class="sr">{verb}</h2><p class="panel-note">{esc(sentence)}</p>{art}</section>'
        )
    return f'<div class="how" id="how">{"".join(out)}</div>'


# ------------------------------------------------------------------------- picker, about, footer
def taste_statement_html() -> str:
    return (
        '<h2 class="statement" id="taste">Tell it what you like. It finds the rest.</h2>'
        '<p class="lead taste-lead">Add two or three products. A recommender trained on 1.35 million '
        "reviews turns them into a taste and ranks everything else.</p>"
    )


def about_html() -> str:
    return (
        '<section class="about band" id="about">'
        '<div class="about-text"><h2 class="statement">Built to show its working.</h2>'
        '<p class="big">CartRank is a portfolio project. It ranks 51,814 '
        "office-product listings and shows its working: what it understood, what it dropped, and why "
        "each result is here.</p>"
        '<p class="note">Titles, prices and photos come from Amazon listings. Parsing runs on free '
        "language-model plans, so when a daily limit is reached it falls back to a simpler parser and "
        "says so.</p></div>"
        '<ul class="facts">'
        "<li><b>Data</b>Amazon Reviews 2023, Office Products; Amazon ESCI shopping queries</li>"
        "<li><b>Models</b>bge-small fine-tuned encoder, MiniLM cross-encoder, a two-tower recommender</li>"
        f'<li><b>Code</b><a href="{REPO}" target="_blank" rel="noopener">github.com/abhibarnwal707/CartRank</a></li>'
        "</ul>"
        "</section>"
    )


def footer_html() -> str:
    return (
        '<div class="close" role="contentinfo">'
        '<div class="close-top"><h2 class="statement">Ask it something awkward.</h2>'
        '<a class="button" href="#top">Rank it</a>'
        f'<ul class="close-links"><li><a href="{REPO}" target="_blank" rel="noopener">GitHub</a></li>'
        f'<li><a href="{REPO}#readme" target="_blank" rel="noopener">Readme</a></li></ul></div>'
        '<p class="credits">Product photos belong to their sellers and are shown from Amazon '
        "listings. Dataset: Amazon Reviews 2023 (McAuley Lab). Queries: Amazon ESCI.</p>"
        '<p class="giant" aria-hidden="true">CartRank</p></div>'
    )
