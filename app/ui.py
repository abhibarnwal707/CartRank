"""HTML builders for the CartRank board. Pure functions: data in, markup out.

Every string that comes from the catalog, the model or the visitor goes through `esc`, because the
markup is rendered with unsafe HTML enabled.
"""

from __future__ import annotations

import html
import math
import re

REPO = "https://github.com/abhibarnwal707/CartRank"
CATALOG = "51,814"
IMG = "app/static/img"  # served by Streamlit static file serving (.streamlit/config.toml)


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def money(x: float) -> str:
    return f"${x:,.2f}" if x != int(x) else f"${int(x):,}"


def _img(url: str | None, cls: str = "", lazy: bool = True) -> str:
    if not url:
        return ""
    load = ' loading="lazy"' if lazy else ""
    c = f' class="{cls}"' if cls else ""
    return f'<img{c} src="{esc(url)}" alt=""{load}>'


# ------------------------------------------------------------------------------------- top bar
# Three bars, longest to shortest: a ranked list as a mark. Spans, because st.html strips SVG.
MARK = '<span class="mark" aria-hidden="true"><i class="m1"></i><i class="m2"></i><i class="m3"></i></span>'


def bar_html() -> str:
    return (
        '<header class="bar" id="top">'
        f'<a class="brand" href="#top">{MARK}<span>CartRank</span></a>'
        '<nav aria-label="Sections"><a href="#results">Results</a><a href="#picker">Recommend</a>'
        '<a href="#how">How it ranks</a><a href="#numbers">Numbers</a></nav>'
        f'<a class="btn ink" href="{REPO}" target="_blank" rel="noopener">Code</a></header>'
    )


# ---------------------------------------------------------------------------------------- wall
# Where the ten picks sit on the wall (row, column), rank order; columns 1-3 show on a phone,
# column 0 is cropped by the bleed and the panel covers the middle of rows 3-5 on a desktop.
PICK_AT = [(1, 1), (0, 3), (2, 1), (0, 5), (0, 8), (1, 11), (0, 10), (2, 10), (3, 1), (4, 11)]
PHOTO_AT = {(0, 0): "pencil-burst", (3, 11): "stapler", (1, 9): "scissors", (4, 0): "name-tags"}
FILLS = ["pale", "paper", "ground", "pale", "blue", "paper", "ground", "red", "pale", "ink"]


def wall_html(
    picks: list[str | None], catalog: list[str | None], rows: int = 6, per_row: int = 15
) -> str:
    """The catalog as a wall of specimens. The ten picks sit on white with their rank; the rest of
    the catalog stays faint on coloured tiles; four found photographs set the scene."""
    pick_at = {pos: (i + 1, url) for i, (pos, url) in enumerate(zip(PICK_AT, picks, strict=False))}
    chosen = set(picks)
    pool = [u for u in catalog if u and u not in chosen] or [u for u in picks if u]
    k = 0
    out = []
    for r in range(rows):
        tiles = []
        for c in range(per_row):
            if (r, c) in pick_at:
                rank, url = pick_at[(r, c)]
                tiles.append(
                    f'<span class="t pick">{_img(url, lazy=False)}<b class="rk">{rank}</b></span>'
                )
            elif (r, c) in PHOTO_AT:
                tiles.append(
                    f'<span class="t photo">{_img(f"{IMG}/{PHOTO_AT[(r, c)]}.jpg")}</span>'
                )
            else:
                fill = FILLS[(r * 7 + c * 3) % len(FILLS)]
                inner = ""
                if fill in ("pale", "paper", "ground") and pool:
                    inner = _img(pool[k % len(pool)])
                    k += 1
                tiles.append(f'<span class="t {fill}">{inner}</span>')
        out.append(f'<div class="wrow{" odd" if r % 2 else ""}">{"".join(tiles)}</div>')
    return f'<div class="wall" aria-hidden="true">{"".join(out)}</div>'


def title_html() -> str:
    return '<h1 class="h1"><span>Ask in a sentence.</span> <span>Get ten, ranked.</span></h1>'


# ------------------------------------------------------------------------------- understanding
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


def toks_html(intent) -> str:
    return "".join(f'<span class="tok {k}">{esc(t)}</span>' for k, t in tokens_for(intent))


def funnel_stages(resp) -> list[tuple[str, int, str]]:
    """(label, count, kind) from the counts the pipeline recorded for this very query."""
    c = resp.debug.get("candidates", {})
    cat = c.get("catalog_size", 51814)
    if resp.route_used == "similar_to":
        return [
            ("in the catalog", cat, "base"),
            ("nearest neighbours", c.get("neighbours_pool", 0), "mid"),
            ("pass your filters", c.get("after_filter", 0), "mid"),
            ("shown", len(resp.results), "end"),
        ]
    return [
        ("in the catalog", cat, "base"),
        ("candidates", c.get("fused", 0), "mid"),
        ("pass your filters", c.get("after_filter", 0), "mid"),
        ("re-read closely", c.get("reranked", 0), "mid"),
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
            f'<li class="stage {kind}"><b>{n:,}</b><span class="stage-text">{esc(label)}</span>'
            f'<span class="stage-bar"><span style="--w:{w:.3f}"></span></span></li>'
        )
    return f'<ol class="flow" aria-label="How the list was narrowed">{"".join(rows)}</ol>'


def hood_html(resp) -> str:
    """The parse, the timings and the model that parsed it."""
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
    return f'<dl class="hood">{rows}</dl>'


# ------------------------------------------------------------------------- index with preview
def _relative(scores: list[float | None]) -> list[float]:
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return [1.0 for _ in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [0.1 + 0.9 * ((s - lo) / span) if s is not None else 0.1 for s in scores]


def _short(part: str) -> str:
    """Trim the templated why-parts to chip size without changing what they say."""
    part = re.sub(r"^price \$[\d.,]+ \(at most (\$[\d.,]+)\)$", r"Under \1", part)
    part = re.sub(r"^price \$[\d.,]+ \(at least (\$[\d.,]+)\)$", r"From \1", part)
    part = re.sub(r"^no '(.+)' mentioned$", r"No \1", part)
    return part


def chips_html(why: str) -> str:
    """The why-line as short chips: the constraint checks first, the keyword echo only if alone."""
    parts = [p.strip() for p in why.split(";") if p.strip()]
    checks = [p for p in parts if not p.startswith("matches")]
    return "".join(f'<span class="chip">{esc(_short(p))}</span>' for p in (checks or parts)[:3])


def row_html(rank: int, r, rel: float) -> str:
    """One ranked row. Its preview sits in the shared frame and shows while the row is pointed at
    (the first row's shows by default)."""
    store = f"{esc(r.store)} · " if r.store else ""
    return (
        f'<li class="irow" tabindex="0" data-price="{r.price:.2f}" data-rank="{rank}">'
        f'<span class="ir">{rank}</span>{_img(r.image_url, "thumb")}'
        f'<span class="it">{esc(r.title[:110])}</span><span class="ip">{money(r.price)}</span>'
        '<div class="pv">'
        f'<div class="pv-ph">{_img(r.image_url)}<b class="rk">{rank}</b></div>'
        '<div class="pv-meta">'
        f'<p class="pv-price">{money(r.price)}</p>'
        f'<p class="pv-title">{esc(r.title[:140])}</p>'
        f'<p class="meta">{store}{r.rating:.1f} of 5 · {r.rating_number:,} ratings</p>'
        f'<div class="chips">{chips_html(r.why)}</div>'
        '<div class="score" role="img" aria-label="Score relative to the others in this list">'
        f'<span style="--w:{rel:.3f}"></span></div></div></div></li>'
    )


def index_html(
    results, heading: str, line: str = "", toks: str = "", anchor: str = "results"
) -> str:
    """Rows of text are the way through the results; the pointed row's product fills one frame."""
    if not results:
        return (
            f'<section class="tile index-tile" id="{anchor}"><h2 class="h2">{esc(heading)}</h2>'
            '<p class="note">Nothing matched every rule. Try fewer constraints.</p></section>'
        )
    rel = _relative([r.score for r in results])
    rows = "".join(row_html(i, r, rel[i - 1]) for i, r in enumerate(results, 1))
    toks = f'<div class="toks">{toks}</div>' if toks else ""
    return (
        f'<section class="tile index-tile" id="{anchor}" aria-label="{esc(heading)}">'
        f'<div class="index-head"><h2 class="h2">{esc(heading)}</h2>'
        f'<p class="q">{line}</p>{toks}</div>'
        f'<div class="index"><ol class="rows">{rows}</ol><div class="frame"></div></div></section>'
    )


def results_html(resp, example: bool = False) -> str:
    line = f"Example: “{esc(resp.query)}”" if example else f"“{esc(resp.query)}”"
    notes = "".join(f' <span class="note-inline">{esc(n)}</span>' for n in resp.notes)
    return index_html(resp.results, "Your ten", line + notes, toks_html(resp.intent))


def recs_html(results) -> str:
    return index_html(results, "In your taste", "Nearest to your picks first.", anchor="recs")


# -------------------------------------------------------------------------------------- picker
def picker_intro_html() -> str:
    return (
        '<div class="tile blue picker-intro" id="picker">'
        f'<div class="cover">{_img(f"{IMG}/stapler.jpg")}</div>'
        '<h2 class="h2">Tell it what you like.</h2>'
        '<p class="lead">Pick up to three. It finds more.</p></div>'
    )


def mini_html(title: str, image_url: str | None, price: float | None = None) -> str:
    p = f'<p class="mini-price">{money(price)}</p>' if price is not None else ""
    return (
        f'<div class="mini"><div class="mini-ph">{_img(image_url)}</div>{p}'
        f'<p class="mini-title">{esc(title[:60])}</p></div>'
    )


def slot_html(n: int) -> str:
    return f'<div class="mini slot"><div class="mini-ph"><b class="rk">{n}</b></div></div>'


# --------------------------------------------------------------------------------- how it ranks
HOW = [
    (
        "pale",
        "handwriting",
        "First it reads.",
        "A small language model (gpt-oss-20b on Groq, with a fallback) turns your sentence into a "
        "query: search words, price range, minimum rating, brand, words to require or exclude. With "
        "no model reachable, rules still catch prices and ratings.",
    ),
    (
        "blue",
        "pencil-tins",
        "Then it gathers.",
        "BM25 keyword search and a bge-small encoder fine-tuned on Amazon's ESCI shopping labels "
        "each return their best 200. Reciprocal-rank fusion merges the two lists.",
    ),
    (
        "ink",
        "video:pencil-sharpener",
        "Then it cuts.",
        "Price, rating and brand are hard filters: a product that breaks one is removed, not just "
        "ranked lower. Excluded words remove any listing that mentions them.",
    ),
    (
        "red",
        "pencil-row",
        "Then it orders.",
        "A MiniLM cross-encoder reads your query next to each of the best 50, scores every pair, "
        "and the scores set the final order. Each result gets a short reason.",
    ),
]


def field_html(i: int, resp) -> str:
    """One of the four fields: a found photograph, the same query card one step further along, and
    the step's clause along the bottom."""
    fill, photo, clause, _ = HOW[i]
    c = resp.debug.get("candidates", {}) if resp else {}
    if i == 0:
        card = f'<p class="card-q">{esc(resp.query) if resp else "black gel pens under $10"}</p>'
    elif i == 1:
        card = (
            '<p class="card-n"><b>200</b> + <b>200</b></p>'
            f'<p class="card-sub">{c.get("fused", 300):,} after merging</p>'
        )
    elif i == 2:
        toks = toks_html(resp.intent) if resp else ""
        card = (
            f'<div class="toks">{toks}</div>'
            f'<p class="card-sub">{c.get("after_filter", 0):,} pass</p>'
        )
    else:
        thumbs = "".join(
            f'<span class="ct">{_img(r.image_url)}<b class="rk">{n}</b></span>'
            for n, r in enumerate((resp.results if resp else [])[:3], 1)
        )
        card = f'<div class="card-top">{thumbs}</div>'
    head = '<h2 class="h2 how-h" id="how">How it ranks</h2>' if i == 0 else ""
    if photo.startswith("video:"):
        v = f"app/static/video/{photo[6:]}"
        media = (
            f'<video src="{v}.webm" poster="{v}-poster.jpg" autoplay muted loop playsinline '
            'preload="metadata" aria-label="A pencil being sharpened"></video>'
        )
    else:
        media = _img(f"{IMG}/{photo}.jpg")
    return (
        f'<div class="field {fill}">{head}<div class="field-ph">{media}</div>'
        f'<div class="card">{card}</div><p class="clause">{clause}</p></div>'
    )


def how_more(i: int) -> str:
    return HOW[i][3]


# -------------------------------------------------------------------------------------- numbers
STATS = [
    (
        "red",
        "+28%",
        "ranking quality over keyword search",
        "NDCG@10 on the Amazon ESCI dev set, full catalog: keyword search (BM25) 0.317, CartRank's "
        "hybrid retrieval plus cross-encoder 0.407. Test-set numbers are pending.",
    ),
    (
        "ink",
        "92%",
        "of result slots meet every stated rule",
        "On 21 queries with two or more constraints: 27.6% of top-10 slots met them all with plain "
        "search, 92.4% with the parser and hard filters. A small sample; read it as indicative.",
    ),
    (
        "blue",
        "51,814",
        "office products, each with a price",
        "Amazon Reviews 2023, Office Products: every listing with a price. The recommender learned "
        "from 1.35 million reviews in the training split.",
    ),
]


# Before and after, as measured (results/search_baselines.md, results/query_layer.md).
STAT_BARS = {
    0: [("Keyword search", 0.3172, "0.317"), ("CartRank", 0.4070, "0.407")],
    1: [("Plain search", 0.276, "27.6%"), ("CartRank", 0.924, "92.4%")],
}


def stat_html(i: int) -> str:
    fill, fig, cap, _ = STATS[i]
    bars = STAT_BARS.get(i)
    chart = ""
    if bars:
        top = max(v for _, v, _ in bars)
        chart = (
            '<div class="sbars">'
            + "".join(
                f'<div class="sbar{" now" if n else ""}"><span class="sb-l">{esc(label)}</span>'
                f'<span class="sb-t"><span style="--w:{v / top:.3f}"></span></span>'
                f'<span class="sb-v">{esc(text)}</span></div>'
                for n, (label, v, text) in enumerate(bars)
            )
            + "</div>"
        )
    return f'<div class="stat {fill}"><p class="fig">{fig}</p>{chart}<p class="cap">{cap}</p></div>'


def numbers_head_html() -> str:
    return '<h2 class="h2 num-h" id="numbers">The numbers</h2>'


def stat_more(i: int) -> str:
    return STATS[i][3]


# --------------------------------------------------------------------------------------- footer
def footer_html() -> str:
    fills = ["red", "blue", "ink", "pale", "red", "blue", "ink", "pale"]
    heights = [100, 78, 64, 88, 100, 70, 84, 94]
    blocks = "".join(
        f'<span class="blk {f}" style="--h:{h}%">{esc(ch)}</span>'
        for ch, f, h in zip("CartRank", fills, heights, strict=True)
    )
    return (
        '<div class="close" role="contentinfo">'
        f'<div class="blocks" aria-label="CartRank">{blocks}</div>'
        '<div class="close-row"><p class="small">A portfolio project. Amazon Reviews 2023 and '
        "Amazon ESCI. Product photos belong to their sellers; scene photos are CC0 from the "
        "WordPress Photo Directory; pencil-sharpener clip by Aerra Carnicom, CC BY-SA 4.0, via "
        "Wikimedia Commons.</p>"
        f'<a class="btn ink" href="{REPO}" target="_blank" rel="noopener">Code on GitHub</a></div>'
        "</div>"
    )
