# Search baselines (ESCI dev)

Dev split: 2,089 queries (10% of train query_ids, split by query). Test split not used. NDCG@10 uses graded gains E=1.0, S=0.1, C=0.01, I=0; Recall@100 counts Exact as relevant (queries with no Exact excluded); MRR@10 is the first Exact result.

## Rerank mode (standard ESCI Task 1): rank each query's judged products only

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.7011 | - | 1.0000 | 0.7545 |
| Dense, bge-small off-the-shelf | 0.7261 | +0.0249 | 1.0000 | 0.7954 |
| Hybrid RRF (BM25 + dense) | 0.7215 | -0.0046 | 1.0000 | 0.7862 |

Recall@100 is trivially 1.0 here: each query has about 20 judged products, so all of them fit in the top 100. Hybrid RRF parameters were tuned on full-corpus NDCG only, which is why hybrid does not beat dense in this mode.

## Full-corpus retrieval mode: retrieve from all US small-set products

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.3172 | - | 0.5750 | 0.4680 |
| Dense, bge-small off-the-shelf | 0.3361 | +0.0190 | 0.6152 | 0.5049 |
| Hybrid RRF (BM25 + dense) | 0.3620 | +0.0259 | 0.6482 | 0.5268 |

Unjudged products count as non-relevant, so full-corpus scores underestimate true quality and are not comparable to rerank-mode scores.

## Hybrid tuning (dev, full-corpus NDCG@10)

Best: k=10, depth=200 (NDCG@10 0.3620).

| k | depth | NDCG@10 | Recall@100 |
|---|---|---|---|
| 10 | 50 | 0.3596 | 0.6165 |
| 10 | 100 | 0.3615 | 0.6437 |
| 10 | 200 | 0.3620 | 0.6482 |
| 30 | 50 | 0.3561 | 0.6165 |
| 30 | 100 | 0.3585 | 0.6437 |
| 30 | 200 | 0.3593 | 0.6497 |
| 60 | 50 | 0.3543 | 0.6165 |
| 60 | 100 | 0.3552 | 0.6437 |
| 60 | 200 | 0.3557 | 0.6497 |
| 100 | 50 | 0.3538 | 0.6165 |
| 100 | 100 | 0.3539 | 0.6437 |
| 100 | 200 | 0.3540 | 0.6479 |

## Latency and index size

Latency is one full-corpus query at a time on the local MacBook Air (query encoding on MPS for dense), 200 dev queries, top-100, 482,105 products.

| System | p50 ms | p95 ms |
|---|---|---|
| bm25 | 1.4 | 2.7 |
| dense_base | 12.8 | 19.2 |
| hybrid_base | 14.9 | 19.4 |

BM25 index: build 12 s, 175 MB on disk. Dense vectors: 384-dim float32, 741 MB (FAISS flat inner product).

## Ablation: product text (BM25)

| Text | NDCG@10 rerank | NDCG@10 full | Recall@100 full |
|---|---|---|---|
| title | 0.6966 | 0.3181 | 0.5574 |
| title + brand + color | 0.6999 | 0.3250 | 0.5619 |
| title + brand + color + bullets (default) | 0.7011 | 0.3172 | 0.5750 |
| title + brand + color + bullets + description | 0.7031 | 0.3192 | 0.5823 |

## Ablation: product text and query prefix (dense, bge-small)

Rerank mode only (the corpus is the dev queries' judged products).

| Text variant | Query prefix | NDCG@10 | MRR@10 |
|---|---|---|---|
| title | prefix | 0.7237 | 0.7878 |
| tbc | prefix | 0.7267 | 0.7995 |
| tbcb | prefix | 0.7261 | 0.7954 |
| tbcb | no_prefix | 0.7261 | 0.7928 |
| tbcbd | prefix | 0.7256 | 0.7925 |
