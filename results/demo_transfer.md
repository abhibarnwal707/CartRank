# Demo catalog transfer check (Office_Products, 50 queries)

Precision@10 on a small sanity check, **not** a headline metric. The ESCI-trained models are applied to a catalog with no relevance labels; the owner hand-labelled the top-10 of each system (blind: products were pooled, shuffled and unlabelled by system).

Final system: hybrid (BM25 + `ft` dense, RRF k=10, depth=200) + cross-encoder rerank of the top 50.

| System | Precision@10 | Δ vs BM25 |
|---|---|---|
| BM25 | 0.594 | - |
| Final | 0.646 | +0.052 |

Paired bootstrap over the 50 queries (10,000 resamples, seed 0): the mean difference has a 95% interval of [+0.014, +0.096]. Per query, the final system is better on 16, tied on 27 and worse on 7.

## By query type

| Type | Queries | BM25 P@10 | Final P@10 |
|---|---|---|---|
| attribute | 15 | 0.553 | 0.627 |
| brand | 10 | 0.640 | 0.640 |
| negation | 5 | 0.260 | 0.220 |
| short | 10 | 0.860 | 0.990 |
| typo | 4 | 0.100 | 0.200 |
| use_case | 6 | 0.783 | 0.783 |

Queries: 50; labelled products: 823. Binary labels (1 relevant, 0 not). Per-type groups have 4 to 15 queries, so treat them as indicative only.
