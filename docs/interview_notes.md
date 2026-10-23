# Interview notes

One entry per headline number: sample size, what it measures, what it does not, and the single
biggest failure mode found. Test-split numbers are **pending** (the locked runs have not been run);
until then the figures below are dev numbers and say so. Fill the `pending` cells from
`results/final_test.md` only.

## 1. Search: NDCG@10 from BM25 to the full pipeline (ESCI)

- **Numbers:** dev, full-corpus 0.3172 (BM25) to 0.4070 (hybrid with fine-tuned dense + cross-encoder);
  rerank mode 0.7011 to 0.7352. Test: *pending*.
- **Sample:** dev 2,089 queries (10% of train queries, split by query); test 8,956 queries; about 20
  judged products per query.
- **Measures:** how well graded human labels (Exact 1.0, Substitute 0.1, Complement 0.01) are ranked
  at the top 10, from Amazon's own ESCI benchmark.
- **Does not measure:** relevance of products nobody judged (full-corpus mode counts them as wrong, so
  it underestimates), other catalogs, or user satisfaction. Rerank mode only orders a ~20-item
  candidate list, so it is easy and its Recall@100 is trivially 1.0.
- **Biggest failure mode:** on 22.4% of dev queries the final system scores below BM25. In a hand-labelled
  sample of 30, about half were near-ties, and the real failures were negation ("without drilling"),
  dropped attributes (colour), numeric specs and ambiguous short queries. Dense models blur exact
  tokens and constraints, which is why the parser adds hard filters.
- **Honest caveat:** fine-tuning the encoder helped the retriever (dense 0.3361 to 0.3542) but the final
  pipeline gained only 0.4052 to 0.4070, within noise, because the cross-encoder dominates.

## 2. Recommender: two-tower vs BPR, full-ranking HR@10 and NDCG@10

- **Numbers:** dev HR@10 0.0462 (two-tower ID + text) vs 0.0260 (BPR) vs 0.0269 (ItemKNN); NDCG@10
  0.0267 vs 0.0147 vs 0.0178. Test: *pending*.
- **Sample:** 223,308 users, 77,551 items, one held-out interaction per user (temporal leave-last-out).
- **Measures:** whether the user's next purchase-proxy item appears in the top 10 of all 77,551 items,
  with no sampled negatives (sampled metrics are known to mislead; Krichene and Rendle, 2020).
- **Does not measure:** online behaviour, diversity, or purchases (reviews are a proxy). Absolute values
  are small because the catalog is large and the task hard; the comparison between models is the point.
- **Biggest failure mode:** cold items. Items with fewer than 5 train interactions: ItemKNN HR@10 0.0109,
  two-tower about 0.0005, BPR 0.0002. The text embedding lifted overall ranking but did not help cold
  items. Untested hypothesis: untrained random ID embeddings swamp the text signal; ID dropout during
  training is the obvious experiment.
- **Choices at grid edges:** the best temperature (0.1) and the best ItemKNN k (200) were the largest
  tried, so they may be under-tuned.

## 3. Query parser: constraint-intent exact match and constraint satisfaction

- **Numbers:** held-out parse accuracy 0.42 (regex) to 0.72 (LLM, prompt v1) to 0.92 (prompt v2). Share of
  top-10 results satisfying every stated constraint: 27.6% (plain search) to 92.4% (parser + filters).
- **Sample:** 50 held-out queries, written before any model output on them was seen, run once per frozen
  prompt; 21 multi-constraint queries for the end-to-end figure, only 6 of them held-out
  (13.3% to 75.0%).
- **Measures:** whether the model recovers route, price range, brand, rating, must-have and exclude terms
  exactly, and whether filters make results obey those constraints.
- **Does not measure:** answer relevance. Satisfying a constraint is not the same as being a good
  result. Must-have and exclude are judged by keyword rules that the filter also applies, so for those it
  is a mechanism check. Both query sets were written by the assistant and have not been reviewed by the
  owner.
- **Biggest failure mode:** prompt v1 over-used the "recommend" route and left brands empty; v2 fixed that
  but was written after reading v1's errors on the dev set, which is why the held-out set matters.
  Operationally, Groq's free tier (1,000 requests and 200K tokens per day) is the bottleneck.

## 4. Serving: memory on a free tier

- **Numbers:** resident memory after loading everything and running 50 queries is 1,334 MB on a laptop
  (`results/serving_footprint.md`) and 1,421 MB on Streamlit Community Cloud (`results/deployed_memory.md`);
  budget 2,000 MB, platform ceiling about 2.7 GB.
- **Measures:** the process's memory with all models and indexes loaded, with the LLM off.
- **Does not measure:** concurrent users (memory per session), or anything about speed. Latency is
  deliberately not reported: the demo runs on free tiers (Streamlit, Groq, OpenRouter), where timings are
  not meaningful.
- **Biggest failure mode:** the cross-encoder batch size. At batch 64 memory grew to 2,406 MB over 50
  queries; batch 4 holds it at 1,421 MB with identical rankings (`results/ce_batch_memory.md`). Found by
  measuring, not by reading code.
- **Operational limit worth mentioning:** Groq's free tier allows about 200K tokens per day, roughly 270
  query parses at the measured prompt size, so the app caches repeated queries and falls back to a free
  OpenRouter model and then to regex-only parsing.

## Questions to be ready for

- Why is the final gain from fine-tuning so small? (The cross-encoder rescues weak first-stage ranking.)
- Why not an agent? (Latency, free-tier cost, and the other projects already show agents.)
- Why do sampled metrics mislead? (Item popularity makes sampled negatives easy.)
- What would you do next? (ID dropout for cold items; fine-tune the cross-encoder; learning-to-rank.)
