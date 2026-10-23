# Serving latency and memory

**Measured locally** on local Darwin arm64, CPU, Python 3.12.15; **not yet measured on Streamlit Cloud** (the deployed numbers go in a separate section once the app is live). 50 scripted queries (`evals/queries/demo_queries.jsonl`) through the real pipeline in a fresh process: parse, hybrid retrieve, hard filter, cross-encoder rerank of the top 50, templated explanation. Encoder and cross-encoder run on CPU.

Budgets (playbook): RSS <= 2,000 MB, cold start <= 90 s, p95 <= 3 s per query.

| Mode | Cold start s (load + first query) | RSS after load MB | RSS after 50 queries MB | p50 ms | p95 ms | max ms | LLM answered |
|---|---|---|---|---|---|---|---|
| LLM off (regex-only) | 3.6 + 2.6 | 1,129 | 1,420 | 216 | 256 | 338 | 0/50 |
| LLM parser on (INVALID run) | 3.7 + 3.1 | - | - | - | - | - | only 8/50: Groq daily token cap hit |

Cold start here is component load plus the first query (which loads the cross-encoder lazily). It excludes downloading artifacts from the Hub and starting Streamlit, which the deployed measurement will add.

**The LLM-on run is invalid and its numbers are not reported.** Only 8 of 50 queries received an LLM answer; the rest fell back to regex-only parsing because Groq returned HTTP 429 on its **daily token limit** (error text: "tokens per day (TPD): Limit 200000, Used 199306"), exhausted earlier the same day by evaluation runs. The end-to-end LLM-on latency is pending a re-run after the quota refills. Measured separately: the parser alone (Groq only, held-out set) had p50 0.78 s and p95 1.32 s (`results/query_layer.md`), and the retrieval pipeline above takes p95 256 ms.

## Per-stage latency (ms)

LLM off:

| Stage | p50 | p95 |
|---|---|---|
| bm25 | 0.4 | 0.6 |
| dense | 6.2 | 6.5 |
| filter | 0.0 | 0.0 |
| fuse | 0.1 | 0.1 |
| parse | 0.0 | 0.0 |
| rerank | 208.7 | 248.7 |

## Cross-encoder batch size vs memory (LLM off, 50 queries)

Batch 64 made resident memory grow well past the budget while queries ran; smaller batches fix that without changing the ranking (the top-10 order was identical on 25 of 25 queries between batch 64 and batch 4). The app uses batch 4.

| CE batch | RSS after load MB | RSS after 50 queries MB | p50 ms | p95 ms |
|---|---|---|---|---|
| 2 | 1,130 | 1,373 | 251 | 296 |
| 4 | 1,130 | 1,421 | 221 | 261 |
| 8 | 1,131 | 1,560 | 210 | 254 |
| 16 | 1,116 | 1,743 | 221 | 258 |
| 64 | 1,128 | 2,406 | 248 | 307 |
