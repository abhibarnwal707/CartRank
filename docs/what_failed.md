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
  rankings and slightly lower latency. Found by measuring, not by reading code.
- Phase 6: I ran `python -m cartrank.eval.latency --help` to see its usage; the script had no argument
  parser, treated `--help` as a mode and started a real 50-query run alongside the latency measurements,
  contaminating them. I stopped everything, re-ran cleanly, and added mode validation.
