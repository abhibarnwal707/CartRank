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
    html = ui.results_html(Response(evil, QueryIntent(text=evil), [r], "search"))
    with_exclude = ui.results_html(
        Response(evil, QueryIntent(text=evil, exclude=[evil]), [r], "search"), example=True
    )
    for out in (html, with_exclude):
        tags, attrs = _tags_and_attrs(out)
        assert "script" not in tags
        assert not any(a.startswith("on") for a in attrs)
    assert "&lt;script&gt;" in html  # shown as text, not run


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


def test_rail_shows_the_sentence_and_tokens_and_results_mark_examples():
    html = ui.results_html(_resp(), example=True)
    assert "black gel pens" in html and 'class="tok need"' in html and 'class="tok skip"' in html
    assert "Example for" in html
    assert "Results for" in ui.results_html(_resp(), example=False)


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


def test_rows_are_in_grid_column_order_and_carry_rank_and_price():
    html = ui.results_html(_resp())
    assert html.count('class="pcard') == 3 and 'class="pcard first"' in html
    row = html.split("<article")[1]
    order = [
        row.index(c) for c in ('class="photo"', 'class="price"', 'class="title"', 'class="chips"')
    ]
    assert order == sorted(order)  # DOM order must follow the columns or CSS grid wraps rows
    assert 'data-price="9.99"' in html and ">$9.99<" in html


def test_hero_strip_holds_at_most_ten_photos_and_skips_missing():
    imgs = [f"https://e.com/{i}.jpg" for i in range(14)] + [None]
    assert ui.strip_html(imgs).count("<img") == 10
    assert ui.strip_html([None, None]).count("<img") == 0
    h = ui.title_html(imgs)
    assert "<h1" in h and "Ask in a sentence." in h and "ranked." in h


def test_how_panels_are_the_four_stages_in_order():
    html = ui.how_html()
    verbs = [v for v in ("Read", "Retrieve", "Filter", "Rerank") if f">{v}</h2>" in html]
    assert verbs == ["Read", "Retrieve", "Filter", "Rerank"]
    assert [html.index(f">{v}</h2>") for v in verbs] == sorted(
        html.index(f">{v}</h2>") for v in verbs
    )


def test_hood_formats_prices_and_hides_nothing_hostile():
    resp = _resp()
    resp.timings_ms = {"parse": 12.0, "total": 99.0}
    html = ui.hood_html(resp)
    assert "price up to</dt><dd>$10<" in html and "10.0" not in html
    assert "minimum rating</dt><dd>4 stars<" in html and "parse 12 ms" in html
