import pandas as pd

from cartrank.data.catalog import build_catalog, catalog_text


def test_catalog_text_layout_and_truncation():
    t = catalog_text("  Blue  Pen ", "Acme", ["a" * 400, "b" * 400])
    assert t.startswith("Blue Pen. Acme. aaaa")
    assert len(t) <= len("Blue Pen. Acme. ") + 500


def test_build_catalog_filters_price_and_title():
    meta = pd.DataFrame(
        {
            "parent_asin": ["a", "b", "c", "d"],
            "title": ["Pen", "Pad", "", "Ink"],
            "store": ["x", None, "y", "z"],
            "price": [1.5, None, 3.0, 0.0],
            "features": [[], [], [], []],
        }
    )
    assert build_catalog(meta).parent_asin.tolist() == ["a"]
