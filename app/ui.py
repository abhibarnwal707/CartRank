"""HTML builders for the CartRank screens. Pure functions: data in, markup out.

Every string that comes from the catalog, the model or the visitor goes through `esc`, because the
markup is rendered with unsafe HTML enabled.
"""

from __future__ import annotations

import html
import math
import re

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
        '<h1 class="title-line"><span class="l">Ask in a sentence.</span>'
        f'<span class="l">Get {strip_html(images)}<span class="sr">ten</span> ranked.</span></h1>'
        '<p class="sub">Describe what you need. Get ten products, ranked, with the reason for each.</p>'
        "</section>"
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


def insight_html(resp) -> str:
    """What the parser understood (constraints as tokens) beside how the list narrowed."""
    intent = resp.intent
    text = intent.text or resp.query
    toks = "".join(f'<span class="tok {k}">{esc(t)}</span>' for k, t in tokens_for(intent))
    notes = "".join(f'<p class="note">{esc(n)}</p>' for n in resp.notes)
    return (
        '<section class="insight"><div class="understood"><p class="label">Understood</p>'
        f'<p class="read">{esc(text)}</p><div class="toks">{toks}</div>{notes}</div>'
        f"{funnel_html(resp)}</section>"
    )


# ----------------------------------------------------------------------------------------- cards
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
    """The why-line as at most two short chips: the constraint checks, not the keyword echo."""
    parts = [p.strip() for p in why.split(";") if p.strip() and not p.strip().startswith("matches")]
    if not parts:
        parts = [why.strip()] if why.strip() else []
    return "".join(f'<span class="chip">{esc(_short(p))}</span>' for p in parts[:2])


def card_html(rank: int, r, rel: float, first: bool) -> str:
    """One ranked product as a card: a large photo with its rank, then price, title and two chips."""
    img = f'<img src="{esc(r.image_url)}" alt="" loading="lazy">' if r.image_url else ""
    store = f"{esc(r.store)} · " if r.store else ""
    return (
        f'<article class="pcard{" first" if first else ""}" data-price="{r.price:.2f}" '
        f'data-rank="{rank}" title="{esc(r.why)}">'
        f'<div class="photo">{img}<span class="badge" aria-label="Rank {rank}">{rank}</span></div>'
        '<div class="pinfo"><div class="pline"><p class="price">'
        f'{money(r.price)}</p><div class="score" role="img" '
        'aria-label="Score relative to the others in this list">'
        f'<span style="--w:{rel:.3f}"></span></div></div>'
        f'<p class="title">{esc(r.title[:100])}</p>'
        f'<p class="meta">{store}{r.rating:.1f} of 5 · {r.rating_number:,}</p>'
        f'<div class="chips">{chips_html(r.why)}</div></div></article>'
    )


def cards_html(results) -> str:
    rel = _relative([r.score for r in results])
    return "".join(card_html(i, r, rel[i - 1], i == 1) for i, r in enumerate(results, 1))


def results_html(resp, example: bool = False) -> str:
    """Understood + narrowing band, then the ten cards, then the raw parse in a disclosure."""
    if not resp.results:
        return '<section class="results"><p class="note">No products found. Try fewer constraints.</p></section>'
    which = (
        f"Example for “{esc(resp.query)}”. Type your own above."
        if example
        else f"Results for “{esc(resp.query)}”"
    )
    return (
        f'<section class="results" aria-label="Ranked results">{insight_html(resp)}'
        f'<p class="caption which">{which}</p>'
        f'<div class="grid">{cards_html(resp.results)}</div>{hood_html(resp)}</section>'
    )


def recs_html(results) -> str:
    if not results:
        return ""
    return (
        '<section class="results"><p class="caption which">Recommended for you, nearest to your '
        f'picks first.</p><div class="grid">{cards_html(results)}</div></section>'
    )


def slot_html(n: int) -> str:
    """An empty pick slot: a blank photo square that a chosen product will fill."""
    return f'<div class="mini slot"><div class="photo"><span class="badge">{n}</span></div></div>'


def mini_html(title: str, image_url: str | None, price: float | None = None) -> str:
    """A small product tile for the picker: photo, short title, optional price."""
    img = f'<img src="{esc(image_url)}" alt="" loading="lazy">' if image_url else ""
    p = f'<p class="price">{money(price)}</p>' if price is not None else ""
    return (
        f'<div class="mini"><div class="photo">{img}</div>{p}'
        f'<p class="title">{esc(title[:60])}</p></div>'
    )


# --------------------------------------------------------------------------------- how it ranks
def _squares(kinds: str) -> str:
    return "".join(f'<i class="sq {k}"></i>' for k in kinds)


def how_html() -> str:
    panels = [
        (
            "field-surface",
            "Read",
            "Your sentence becomes filters: price, rating, brand, what to skip.",
            '<div class="art art-read"><span class="w">black gel pens</span>'
            '<span class="tok skip">no clips</span><span class="tok need">up to $10</span></div>',
        ),
        (
            "field-accent",
            "Retrieve",
            "Keywords and a fine-tuned model each pull their best 200. Merged into one list.",
            f'<div class="art art-merge"><div>{_squares("aabab")}</div><div>{_squares("bbaab")}</div>'
            f'<div class="out">{_squares("abababab")}</div></div>',
        ),
        (
            "field-ink",
            "Filter",
            "Hard filters drop anything that breaks your rules.",
            f'<div class="art art-filter"><div>{_squares("kxkkxkxk")}</div>'
            f"<div>{_squares('kkxkkkxk')}</div><div>{_squares('xkkxkkkk')}</div></div>",
        ),
        (
            "field-surface",
            "Rerank",
            "A cross-encoder reads the best 50 against your query and orders them.",
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


# ---------------------------------------------------------------------------------------- footer
def footer_html() -> str:
    return (
        '<div class="foot-row" role="contentinfo"><p class="caption">A portfolio project. '
        f"{CATALOG} Amazon office products, Amazon Reviews 2023 and ESCI. Photos belong to their "
        "sellers.</p>"
        f'<a class="code-link" href="{REPO}" target="_blank" rel="noopener">Code on GitHub</a></div>'
    )
