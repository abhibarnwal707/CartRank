# Search baselines (ESCI dev)

Dev split: 2,089 queries (10% of train query_ids, split by query). Test split not used. NDCG@10 uses graded gains E=1.0, S=0.1, C=0.01, I=0; Recall@100 counts Exact as relevant (queries with no Exact excluded); MRR@10 is the first Exact result.

## Rerank mode (standard ESCI Task 1): rank each query's judged products only

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.7011 | - | 1.0000 | 0.7545 |
| Dense, bge-small off-the-shelf | 0.7261 | +0.0249 | 1.0000 | 0.7954 |
| Dense, bge-small fine-tuned on ESCI | 0.7351 | +0.0091 | 1.0000 | 0.7875 |
| Hybrid RRF (BM25 + fine-tuned dense) | 0.7283 | -0.0068 | 1.0000 | 0.7915 |
| Hybrid-FT + cross-encoder rerank (full-corpus: top-50 of hybrid) | 0.7352 | +0.0069 | 1.0000 | 0.8044 |

Recall@100 is trivially 1.0 here: each query has about 20 judged products, so all of them fit in the top 100. Hybrid RRF parameters were tuned on full-corpus NDCG only, so hybrid can trail the best single retriever in this mode. The cross-encoder row is the same for any first stage here: it orders all judged candidates itself.

## Full-corpus retrieval mode: retrieve from all US small-set products

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.3172 | - | 0.5750 | 0.4680 |
| Dense, bge-small off-the-shelf | 0.3361 | +0.0190 | 0.6152 | 0.5049 |
| Dense, bge-small fine-tuned on ESCI | 0.3542 | +0.0181 | 0.6524 | 0.5168 |
| Hybrid RRF (BM25 + fine-tuned dense) | 0.3776 | +0.0233 | 0.6786 | 0.5434 |
| Hybrid-FT + cross-encoder rerank (full-corpus: top-50 of hybrid) | 0.4070 | +0.0295 | 0.6786 | 0.5821 |

Unjudged products count as non-relevant, so full-corpus scores underestimate true quality and are not comparable to rerank-mode scores.

## Reference: off-the-shelf hybrid (Phase 1) and its rerank

rerank mode:

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| Hybrid RRF (BM25 + off-the-shelf dense) | 0.7215 | - | 1.0000 | 0.7862 |
| Hybrid-base + cross-encoder rerank (full-corpus: top-50) | 0.7352 | +0.0137 | 1.0000 | 0.8044 |

full mode:

| System | NDCG@10 | Δ vs previous row | Recall@100 | MRR@10 |
|---|---|---|---|---|
| Hybrid RRF (BM25 + off-the-shelf dense) | 0.3620 | - | 0.6482 | 0.5268 |
| Hybrid-base + cross-encoder rerank (full-corpus: top-50) | 0.4052 | +0.0432 | 0.6482 | 0.5807 |

## Hybrid tuning, fine-tuned encoder (dev, full-corpus NDCG@10)

Best: k=10, depth=200 (NDCG@10 0.3776).

| k | depth | NDCG@10 | Recall@100 |
|---|---|---|---|
| 10 | 50 | 0.3743 | 0.6525 |
| 10 | 100 | 0.3767 | 0.6763 |
| 10 | 200 | 0.3776 | 0.6786 |
| 30 | 50 | 0.3681 | 0.6525 |
| 30 | 100 | 0.3716 | 0.6763 |
| 30 | 200 | 0.3730 | 0.6786 |
| 60 | 50 | 0.3664 | 0.6525 |
| 60 | 100 | 0.3676 | 0.6763 |
| 60 | 200 | 0.3682 | 0.6784 |
| 100 | 50 | 0.3663 | 0.6525 |
| 100 | 100 | 0.3665 | 0.6763 |
| 100 | 200 | 0.3669 | 0.6749 |

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

## Cross-encoder rerank (full-corpus, hybrid first stage)

Model `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M params), input = query + product text truncated to 256 tokens, scored on CPU. In rerank mode the cross-encoder orders every judged candidate, so k does not apply. Latency is the cross-encoder stage alone, 50 dev queries, local MacBook Air CPU.

| k | NDCG@10 | Δ vs hybrid | Recall@100 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 20 | 0.3984 | +0.0364 | 0.6482 | 0.5766 | 105 | 131 |
| 50 | 0.4052 | +0.0432 | 0.6482 | 0.5807 | 279 | 360 |
| 100 | 0.4066 | +0.0446 | 0.6482 | 0.5804 | 496 | 645 |

Working point k=50: most of the gain of k=100 at about half its latency (chosen on dev, not test).

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
