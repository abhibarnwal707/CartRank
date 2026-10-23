# Test split runs

Each test-split evaluation is run once, after all tuning on dev.

| Stage | UTC time | Git state | What |
|---|---|---|---|
| esci-search-1 | 2026-10-23 17:10 | 42c41ff415db (7 uncommitted paths) | 8956 queries: BM25, dense base/FT, hybrid base/FT, both modes |
| esci-search-2 | 2026-10-23 18:06 | 4c9970057d73 (1 uncommitted paths) | 8956 queries: cross-encoder rerank (top-50 / judged) |
