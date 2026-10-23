# Serving footprint (demo catalog, CPU)

Measured locally on Darwin arm64, Python 3.12.15, encoder and cross-encoder on CPU. Resident set size (RSS) via `ps`, cumulative as each component loads.

| Step | RSS MB | Added MB |
|---|---|---|
| Python + libraries imported | 110 | +110 |
| Catalog metadata (51,814 items) | 491 | +381 |
| BM25 index | 661 | +170 |
| Dense vectors (float32, 384-dim) | 737 | +76 |
| Fine-tuned bi-encoder (CPU) | 1,068 | +331 |
| Cross-encoder MiniLM-L6 (CPU) | 1,080 | +13 |
| After running 50 queries | 1,334 | +253 |

Budget: 2,000 MB. Final steady-state RSS 1,334 MB (within budget).

## Artifact sizes on disk (MB)

- catalog.parquet: 47
- ft_vecs.npy: 80
