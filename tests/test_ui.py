import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import ui  # noqa: E402

from cartrank.query.schema import QueryIntent  # noqa: E402
from cartrank.serve.pipeline import Response, Result  # noqa: E402


def _result(title="Black Gel Pens", price=9.99, score=1.0, **kw):
    base = dict(
        asin="A1",
        title=title,
        store="Acme",
        price=price,
        rating=4.5,
        rating_number=1234,
        image_url="https://example.com/a.jpg",
        why="matches 'gel'",
        score=score,
    )
    base.update(kw)
    return Result(**base)


def _resp(**kw):
    intent = QueryIntent(text="black gel pens", price_max=10, exclude=["clips"], min_rating=4)
    results = [_result(score=s, asin=f"A{i}") for i, s in enumerate([3.0, 1.0, -1.0], 1)]
    return Response(
        "black gel pens without clips under $10 4 stars",
        intent,
        results,
        "search",
        debug={
            "candidates": {
                "catalog_size": 51814,
                "fused": 300,
                "after_filter": 130,
                "reranked": 50,
                "shown": 3,
            }
        },
        **kw,
    )


def _tags_and_attrs(markup: str):
    """Parse the markup like a browser would: every element name and every attribute name."""
    from html.parser import HTMLParser

    seen_tags, seen_attrs = [], []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            seen_tags.append(tag)
            seen_attrs.extend(k for k, _ in attrs)

    P().feed(markup)
    return seen_tags, seen_attrs


def test_everything_from_outside_is_escaped():
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    r = _result(title=evil, store=evil, why=evil, image_url='"><script>x</script>')
    resp = Response(evil, QueryIntent(text=evil, exclude=[evil]), [r], "search")
    outs = [
        ui.results_html(resp, example=True),
        ui.wall_html([r.image_url], [r.image_url]),
        ui.field_html(0, resp),
        ui.field_html(2, resp),
        ui.field_html(3, resp),
        ui.hood_html(resp),
        ui.mini_html(evil, r.image_url, 1.0),
    ]
    for out in outs:
        tags, attrs = _tags_and_attrs(out)
        assert "script" not in tags
        assert not any(a.startswith("on") for a in attrs)
    assert "&lt;script&gt;" in outs[0]  # shown as text, not run


def test_tokens_cover_every_constraint_kind():
    i = QueryIntent(
        route="similar_to",
        price_min=5,
        price_max=15,
        min_rating=4.5,
        must_have=["cordless"],
        exclude=["clips"],
    )
    toks = dict((label, kind) for kind, label in ui.tokens_for(i))
    assert toks["similar to"] == "skip" and toks["$5–$15"] == "need"
    assert (
        toks["4.5 stars and up"] == "need"
        and toks["cordless"] == "need"
        and toks["no clips"] == "skip"
    )
    assert ui.tokens_for(QueryIntent(price_max=10)) == [("need", "up to $10")]
    assert ui.tokens_for(QueryIntent(price_min=20)) == [("need", "from $20")]


def test_results_show_query_tokens_and_mark_examples():
    html = ui.results_html(_resp(), example=True)
    assert 'class="tok need"' in html and 'class="tok skip"' in html
    assert "Example:" in html
    assert "Example:" not in ui.results_html(_resp(), example=False)


def test_funnel_uses_the_recorded_counts_in_order():
    stages = ui.funnel_stages(_resp())
    assert [n for _, n, _ in stages] == [51814, 300, 130, 50, 3]
    html = ui.funnel_html(_resp())
    assert "<b>51,814</b>" in html and "in the catalog" in html and "<b>130</b>" in html
    widths = [float(w) for w in __import__("re").findall(r"--w:([\d.]+)", html)]
    assert widths == sorted(widths, reverse=True) and widths[0] == 1.0


def test_relative_scores_span_the_list_and_never_hit_zero():
    rel = ui._relative([3.0, 1.0, -1.0, None])
    assert rel[0] == 1.0 and rel[2] == 0.1 and rel[1] < rel[0] and rel[3] == 0.1
    assert ui._relative([2.0]) == [1.0]


def test_index_rows_carry_rank_price_and_one_preview_each():
    html = ui.results_html(_resp())
    assert html.count('class="irow"') == 3 and html.count('class="pv"') == 3
    row = html.split('<li class="irow"')[1]
    order = [row.index(c) for c in ('class="ir"', 'class="it"', 'class="ip"', 'class="pv"')]
    assert order == sorted(order)
    assert 'data-price="9.99"' in html and ">$9.99<" in html


def test_chips_shorten_the_why_line_without_inventing():
    why = "matches 'black', 'gel'; price $9.99 (at most $10); no 'clips' mentioned"
    html = ui.chips_html(why)
    assert ">Under $10<" in html and ">No clips<" in html and "matches" not in html
    assert ">because you liked X<" in ui.chips_html("because you liked X")


def test_wall_lights_up_the_picks_in_rank_order():
    picks = [f"https://e.com/p{i}.jpg" for i in range(10)]
    catalog = [f"https://e.com/c{i}.jpg" for i in range(30)]
    html = ui.wall_html(picks, catalog)
    assert html.count('class="t pick"') == 10
    assert [int(n) for n in __import__("re").findall(r'class="rk">(\d+)<', html)] == sorted(
        range(1, 11), key=lambda n: ui.PICK_AT[n - 1]
    )
    assert html.count("app/static/img/") == len(ui.PHOTO_AT)


def test_how_fields_are_the_four_steps_in_order():
    clauses = [ui.HOW[i][2] for i in range(4)]
    assert clauses == ["First it reads.", "Then it gathers.", "Then it cuts.", "Then it orders."]
    assert 'id="how"' in ui.field_html(0, _resp()) and 'id="how"' not in ui.field_html(1, _resp())
    assert ui.field_html(3, _resp()).count('class="ct"') == 3


def test_hood_formats_prices_and_timings():
    resp = _resp()
    resp.timings_ms = {"parse": 12.0, "total": 99.0}
    html = ui.hood_html(resp)
    assert "price up to</dt><dd>$10<" in html and "10.0" not in html
    assert "minimum rating</dt><dd>4 stars<" in html and "parse 12 ms" in html


def test_stat_figures_come_from_results_files():
    root = Path(__file__).resolve().parents[1] / "results"
    search = (root / "search_baselines.md").read_text()
    query = (root / "query_layer.md").read_text()
    assert "| BM25 | 0.3172 |" in search and "| 0.4070 |" in search
    assert round((0.4070 - 0.3172) / 0.3172 * 100) == 28 and ui.STATS[0][1] == "+28%"
    assert "| 27.6% | 92.4% |" in query and ui.STATS[1][1] == "92%"
