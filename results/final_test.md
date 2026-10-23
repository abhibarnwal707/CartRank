# Final test results

Each test split was evaluated once (see test_runs.md). Configurations were chosen on dev only. No significance tests were run, so differences of about 0.005 or less should not be read as real.

## Search, ESCI test (8,956 queries)

| System | Full-corpus NDCG@10 | Δ vs previous | Recall@100 | Rerank-mode NDCG@10 | Δ vs previous |
|---|---|---|---|---|---|
| BM25 | 0.3140 | - | 0.5724 | 0.7021 | - |
| Dense, bge-small off-the-shelf | 0.3325 | +0.0184 | 0.6073 | 0.7301 | +0.0280 |
| Dense, fine-tuned on ESCI | 0.3535 | +0.0210 | 0.6474 | 0.7407 | +0.0106 |
| Hybrid RRF (BM25 + fine-tuned dense) | 0.3721 | +0.0186 | 0.6756 | 0.7313 | -0.0094 |
| Hybrid + cross-encoder (top-50) | 0.4023 | +0.0302 | 0.6756 | 0.7379 | +0.0066 |

Full-corpus counts unjudged products as non-relevant and is not comparable to rerank mode. In rerank mode the hybrid scores below the fine-tuned dense retriever alone (as on dev), and the cross-encoder pipeline does not beat dense-only there; the full-corpus ladder is monotonic.

## Recommendation, Office_Products test (223,308 users, full ranking over 77,551 items)

| Model | HR@10 | NDCG@10 | Cold-slice HR@10 (28,803 users) |
|---|---|---|---|
| Popularity | 0.0098 | 0.0057 | 0.0000 |
| ItemKNN | 0.0212 | 0.0139 | 0.0106 |
| BPR-MF | 0.0186 | 0.0103 | 0.0001 |
| Two-tower, ID-only | 0.0303 | 0.0172 | 0.0002 |
| Two-tower, ID + text | 0.0354 | 0.0200 | 0.0004 |

Check: the BPR retrained for the test run scores HR@10 0.0260 / NDCG@10 0.0147 on dev, matching its dev row. Every model scores lower on test than on dev, with the same ordering. Cold items (fewer than 5 train interactions): only ItemKNN retrieves them; both two-tower variants are near zero.
