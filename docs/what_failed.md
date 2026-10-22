# What failed

Running log of things that went wrong, feeding the README.

- Phase 0: `meta_*.jsonl` `images` field is a list of dicts, not a dict as first assumed; the
  first parsing attempt crashed. Fixed in `cartrank.data.reviews._first_image`.
- Phase 1: hybrid RRF tuned on full-corpus NDCG gives dev-best k=10, depth=200, but in rerank mode hybrid
  (0.7215) is slightly below dense alone (0.7261). Tuning target and evaluation mode disagree; reported as is.
- Phase 1: text-variant ablation (title vs +brand/color vs +bullets vs +description) moved NDCG@10 by at
  most 0.007 for BM25 and 0.003 for dense, so product text is not a lever at this scale.
