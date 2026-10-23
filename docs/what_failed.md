# What failed

Running log of things that went wrong, feeding the README.

- Phase 0: `meta_*.jsonl` `images` field is a list of dicts, not a dict as first assumed; the
  first parsing attempt crashed. Fixed in `cartrank.data.reviews._first_image`.
- Phase 1: hybrid RRF tuned on full-corpus NDCG gives dev-best k=10, depth=200, but in rerank mode hybrid
  (0.7215) is slightly below dense alone (0.7261). Tuning target and evaluation mode disagree; reported as is.
- Phase 1: text-variant ablation (title vs +brand/color vs +bullets vs +description) moved NDCG@10 by at
  most 0.007 for BM25 and 0.003 for dense, so product text is not a lever at this scale.
- Phase 2: running the CPU cross-encoder in the same process as the faiss index deadlocked on macOS (two
  OpenMP runtimes); importing faiss after torch segfaulted instead. Fixed by splitting candidate
  generation (faiss, imported first) and cross-encoder scoring into separate processes.
- Phase 2: the fine-tuned encoder improves retrieval on its own (dense full-corpus NDCG@10 0.3361 -> 0.3542) but
  after the cross-encoder rerank the final gain over the off-the-shelf pipeline is only 0.4052 -> 0.4070 (top-50),
  which is within the range I would expect from noise. The cross-encoder does most of the work.
- Phase 2: the keyword heuristic for tagging error-analysis queries (brand in query) fired on 23 of 30 sample
  queries, so it carried no signal; the reasons were labelled by hand instead.
- Phase 6: with the cross-encoder at batch size 64, the app pipeline's resident memory grew from 1.1 GB
  after load to 2.4 GB after 50 queries, over the 2.0 GB budget. Batch size 4 gives 1.4 GB with identical
  rankings. Found by measuring, not by reading code.
- Phase 6: Groq's free tier also has a daily token cap (200,000 tokens/day for gpt-oss-20b, refilling
  at about 2.3 tokens/s). At about 730 tokens per parse that is roughly 270 parses per day, shared by
  the live demo and every evaluation run. Evaluation runs used it up, so parsing silently fell back to
  regex-only for hours. Lesson: budget API tokens before running quota-limited experiments, and keep
  the demo's share (cache repeated queries; fall back to OpenRouter, then regex-only).
- Phase 6: httpx timeouts apply per read, and OpenRouter keeps the connection alive while a slow free
  model works, so a "5 s" timeout was not a total limit and one call could run for tens of seconds.
  Fixed with a wall-clock deadline per call (tested).
- Phase 6: `.gitignore` had a bare `data/` rule meant for the top-level dataset folder, which also matched
  `src/cartrank/data/`. The whole `cartrank.data` package (paths, splits, catalog, loaders) was never
  committed, so GitHub Actions failed (ruff could not classify `cartrank.data` imports) and the Streamlit
  app would have crashed on import. It also hid that package from my local ruff runs, because ruff respects
  `.gitignore`: three lint/format problems sat unseen. Found by cloning the repo fresh and running the CI
  commands. Fixed by anchoring the rules (`/data/`, `/artifacts/`, `/logs/`...).
