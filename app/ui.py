"""HTML builders for the CartRank screens. Pure functions: data in, markup out.

Every string that comes from the catalog, the model or the visitor goes through `esc`, because the
markup is rendered with unsafe HTML enabled.
"""

from __future__ import annotations

import html
import math

REPO = "https://github.com/abhibarnwal707/CartRank"
CATALOG = "51,814"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def money(x: float) -> str:
    return f"${x:,.2f}" if x != int(x) else f"${int(x):,}"


# ------------------------------------------------------------------------------- top bar, titles
MARK = (
    '<svg class="mark" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<rect x="2" y="3" width="20" height="4.5" fill="#a1ffcb"/>'
    '<rect x="2" y="10" width="14" height="4.5" fill="#232323"/>'
    '<rect x="2" y="17" width="8" height="4.5" fill="#232323"/></svg>'
)


def bar_html() -> str:
    return (
        '<header class="bar">'
        f'<a class="brand" href="{REPO}" target="_blank" rel="noopener">{MARK}<span>CartRank</span></a>'
        f'<a class="code-link" href="{REPO}" target="_blank" rel="noopener">Code</a></header>'
    )


def strip_html(images: list[str | None]) -> str:
    """The ten result photos as a one-line strip that sits in the title where 'ten' goes."""
    tiles = "".join(
        f'<span class="tile"><img src="{esc(u)}" alt="" loading="eager"></span>'
        for u in (images or [])[:10]
        if u
    )
    return f'<span class="strip" role="img" aria-label="ten">{tiles}</span>'


def title_html(images: list[str | None]) -> str:
    return (
        '<section class="title-row">'
        '<h1 class="title-line">Ask in a sentence. Get '
        f'{strip_html(images)}<span class="sr">ten</span> ranked.</h1>'
        f'<p class="sub">CartRank reads what you type, narrows {CATALOG} office products to ten, '
        "and says why each one made the list.</p></section>"
    )


def screen_head_html(title: str, sub: str) -> str:
    return (
        f'<section class="title-row"><h1 class="screen-title">{esc(title)}</h1>'
        f'<p class="sub">{esc(sub)}</p></section>'
    )


def label_html(text: str) -> str:
    return f'<p class="label">{esc(text)}</p>'


# --------------------------------------------------------------------------------- understanding
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


def funnel_stages(resp) -> list[tuple[str, int, str]]:
    """(label, count, kind) from the counts the pipeline recorded for this very query."""
    c = resp.debug.get("candidates", {})
    cat = c.get("catalog_size", 51814)
    if resp.route_used == "similar_to":
        return [
            ("in the catalog", cat, "base"),
            ("nearest to the reference item", c.get("neighbours_pool", 0), "mid"),
            ("pass your filters", c.get("after_filter", 0), "mid"),
            ("shown", len(resp.results), "end"),
        ]
    return [
        ("in the catalog", cat, "base"),
        ("candidates: keywords and meaning, merged", c.get("fused", 0), "mid"),
        ("pass your filters", c.get("after_filter", 0), "mid"),
        ("re-read by the cross-encoder", c.get("reranked", 0), "mid"),
        ("shown", len(resp.results), "end"),
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
            f'<li class="stage {kind}"><span class="stage-text"><b>{n:,}</b> {esc(label)}</span>'
            f'<span class="stage-bar"><span style="--w:{w:.3f}"></span></span></li>'
        )
    return f'<ol class="funnel" aria-label="How the list was narrowed">{"".join(rows)}</ol>'


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


def rail_html(resp) -> str:
    """The right rail: what the parser understood, how the list was narrowed, the raw parse."""
    intent = resp.intent
    text = intent.text or resp.query
    toks = "".join(f'<span class="tok {k}">{esc(t)}</span>' for k, t in tokens_for(intent))
    notes = "".join(f'<p class="note">{esc(n)}</p>' for n in resp.notes)
    return (
        '<aside class="rail">'
        '<section class="card"><h2 class="label">What it understood</h2>'
        f'<p class="read">{esc(text)}</p><div class="toks">{toks}</div>{notes}</section>'
        '<section class="card"><h2 class="label">How it narrowed</h2>'
        f"{funnel_html(resp)}</section>"
        f"{hood_html(resp)}</aside>"
    )


# ----------------------------------------------------------------------------------------- rows
def _relative(scores: list[float | None]) -> list[float]:
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return [1.0 for _ in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [0.1 + 0.9 * ((s - lo) / span) if s is not None else 0.1 for s in scores]


def row_html(rank: int, r, rel: float, first: bool) -> str:
    """One ranked product. DOM order follows the grid columns (rank, photo, body, side), which
    CSS grid auto-placement depends on."""
    img = f'<img src="{esc(r.image_url)}" alt="" loading="lazy">' if r.image_url else ""
    store = f"{esc(r.store)} · " if r.store else ""
    return (
        f'<article class="row{" first" if first else ""}" '
        f'data-price="{r.price:.2f}" data-rank="{rank}">'
        f'<div class="rank" aria-label="Rank {rank}">{rank}</div>'
        f'<div class="photo">{img}</div>'
        '<div class="body">'
        f'<p class="title">{esc(r.title[:120])}</p>'
        f'<p class="meta">{store}{r.rating:.1f} of 5 · {r.rating_number:,} ratings</p>'
        f'<p class="why">{esc(r.why)}</p></div>'
        f'<div class="side"><p class="figure">{money(r.price)}</p>'
        '<div class="score" role="img" aria-label="Score relative to the others in this list">'
        f'<span style="--w:{rel:.3f}"></span></div></div>'
        "</article>"
    )


def rows_html(results) -> str:
    rel = _relative([r.score for r in results])
    return "".join(row_html(i, r, rel[i - 1], i == 1) for i, r in enumerate(results, 1))


def results_html(resp, example: bool = False) -> str:
    """The results: the ranked list on the left, the rail that explains it on the right."""
    if not resp.results:
        return '<section class="results"><p class="note">No products found. Try fewer constraints.</p></section>'
    which = (
        f"Example result for “{esc(resp.query)}”. Type your own above."
        if example
        else f"Results for “{esc(resp.query)}”"
    )
    return (
        '<section class="results" aria-label="Ranked results">'
        '<div class="list"><div class="list-head"><h2 class="label">Ranked results</h2>'
        f'<p class="caption">{which}</p></div>{rows_html(resp.results)}'
        '<p class="caption foot">Bars compare each result with the others in this list; they are '
        f"not percentages.</p></div>{rail_html(resp)}</section>"
    )


def recs_html(results) -> str:
    if not results:
        return ""
    return (
        '<section class="results one"><div class="list"><div class="list-head">'
        '<h2 class="label">Recommended for you</h2>'
        f'<p class="caption">{len(results)} products, nearest to your picks first.</p></div>'
        f"{rows_html(results)}</div></section>"
    )


# --------------------------------------------------------------------------------- how it ranks
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
    out = "".join(
        f'<section class="panel {cls}">{art}<div class="panel-text"><h2 class="verb">{verb}</h2>'
        f'<p class="panel-note">{esc(sentence)}</p></div></section>'
        for cls, verb, sentence, art in panels
    )
    return f'<div class="how-grid">{out}</div>'


# ---------------------------------------------------------------------------------- about, footer
def about_html() -> str:
    return (
        '<section class="about"><h2 class="label">About</h2>'
        '<div class="about-grid"><p class="big">CartRank is a portfolio project. It ranks '
        f"{CATALOG} office-product listings and shows its working: what it understood, what it "
        "dropped, and why each result is here.</p>"
        '<ul class="facts">'
        "<li><b>Data</b>Amazon Reviews 2023, Office Products; Amazon ESCI shopping queries</li>"
        "<li><b>Models</b>bge-small fine-tuned encoder, MiniLM cross-encoder, a two-tower recommender</li>"
        "<li><b>Limits</b>Titles, prices and photos come from Amazon listings. Parsing runs on free "
        "language-model plans, so when a daily limit is reached it falls back to a simpler parser and "
        "says so.</li></ul></div></section>"
    )


def footer_html() -> str:
    return (
        '<div class="foot-row" role="contentinfo"><p class="caption">Product photos belong to their '
        "sellers and are shown from Amazon listings. Dataset: Amazon Reviews 2023 (McAuley Lab). "
        "Queries: Amazon ESCI.</p>"
        f'<a class="code-link" href="{REPO}" target="_blank" rel="noopener">Code on GitHub</a></div>'
    )
