# Serving footprint (demo catalog, CPU)

Measured locally on Darwin arm64, Python 3.12.15, encoder and cross-encoder on CPU. **Not yet measured on Streamlit Cloud** (Linux); that is Phase 6. Resident set size (RSS) via `ps`, cumulative as each component loads.

| Step | RSS MB | Added MB |
|---|---|---|
| Python + libraries imported | 102 | +102 |
| Catalog metadata (51,814 items) | 468 | +366 |
| BM25 index | 639 | +171 |
| Dense vectors (float32, 384-dim) | 715 | +76 |
| Fine-tuned bi-encoder (CPU) | 1,043 | +328 |
| Cross-encoder MiniLM-L6 (CPU) | 1,055 | +13 |
| After running 50 queries | 1,732 | +676 |

Budget: 2,000 MB. Final steady-state RSS 1,732 MB (within budget).

Component load time (catalog, BM25 build, vectors, both models; models already cached on disk): 6 s. This is not the full app cold start, which also includes downloading artifacts and starting Streamlit.

## Per-stage latency (ms, one query at a time, 50 demo queries, CPU)

| Stage | p50 | p95 |
|---|---|---|
| BM25 top-200 | 0.3 | 0.6 |
| Dense (encode query + search) | 5.9 | 6.3 |
| Hybrid retrieve (BM25 + dense + RRF) | 6.2 | 6.7 |
| Cross-encoder rerank, top-50 | 245.3 | 321.2 |
| Whole pipeline (retrieve + rerank) | 250.0 | 312.1 |

The cross-encoder row re-ranks BM25's top 50 only (a timing proxy for 50 pairs); the whole-pipeline row is the real hybrid top-50 rerank.

## Artifact sizes on disk (MB)

- catalog.parquet: 47
- ft_vecs.npy: 80
