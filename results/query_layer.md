# Query understanding layer (Phase 5)

One LLM call turns a free-text query into a typed `QueryIntent` (route, text, price range, brand, minimum rating, must-have and exclude terms). A deterministic regex pre-pass runs first and wins on price and rating. If the LLM fails, the app falls back to the regex result with route `search`.

Model: Groq `openai/gpt-oss-20b` (primary), then OpenRouter free models in order (`recursive/flint-v1.0:free`, `nvidia/nemotron-3.5-super-vl:free`, `google/gemma-4-26b-a4b-it:free`). All parser rows below use Groq only, paced under the free-tier request limit.

## Parser accuracy

Fields are scored by exact match (brand case-insensitive; must-have/exclude as sets; hyphens ignored). `text` is mean token F1. *Constraint-intent exact* = every field except text correct. `category` is not scored (no canonical vocabulary).

### Held-out set (50 queries written before any model output on them was seen, run once per frozen prompt): the number to quote

| Run | n | route | price_min | price_max | brand | min_rating | must_have | exclude | text F1 | Constraint-intent exact | LLM answered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex-only (no LLM) | 50 | 0.92 | 1.00 | 0.98 | 0.76 | 1.00 | 0.86 | 0.80 | 0.73 | 0.42 | - |
| LLM prompt v1 | 50 | 0.94 | 1.00 | 1.00 | 0.92 | 1.00 | 0.92 | 0.94 | 0.89 | 0.72 | 50/50 |
| LLM prompt v2 | 50 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.96 | 0.98 | 0.91 | 0.92 | 49/50 |

### Dev set (100 queries, used while writing the prompts, so v2 is optimistic here)

| Run | n | route | price_min | price_max | brand | min_rating | must_have | exclude | text F1 | Constraint-intent exact | LLM answered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex-only (no LLM) | 100 | 0.96 | 1.00 | 0.99 | 0.78 | 0.99 | 0.86 | 0.82 | 0.75 | 0.47 | - |
| LLM prompt v1 | 100 | 0.92 | 1.00 | 1.00 | 0.88 | 0.99 | 0.91 | 0.98 | 0.90 | 0.72 | 100/100 |
| LLM prompt v2 (tuned on this set) | 100 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 0.90 | 0.92 | 98/100 |

When the LLM did not answer (the error was recorded only as `LLMError`; most likely the free-tier rate limit), the query is scored with its regex-only fallback, so the numbers include those failures.

### Held-out, prompt v2, by query group

| Group | n | Constraint-intent exact | text F1 |
|---|---|---|---|
| plain | 6 | 1.00 | 1.00 |
| price | 6 | 1.00 | 1.00 |
| rating | 4 | 1.00 | 1.00 |
| brand | 6 | 0.83 | 1.00 |
| negation | 5 | 1.00 | 1.00 |
| must_have | 5 | 0.60 | 0.61 |
| multi | 6 | 1.00 | 0.91 |
| typo | 4 | 1.00 | 1.00 |
| non_english | 4 | 0.75 | 0.50 |
| route | 4 | 1.00 | 1.00 |

Groups have 4 to 6 queries each; read them as indicative only.

## Constraint satisfaction, end to end

Queries with at least two stated constraints (price, rating, brand, each must-have and exclude term). Top-10 of (a) plain hybrid search + cross-encoder on the raw query vs (b) parser + hard filters. Constraints are the gold labels. Cell = share of the 10 result slots whose product satisfies every constraint (a slot with no result counts as a failure). "Objective" counts only price, rating and brand (catalog metadata).

| Subset | Queries | All: plain | All: parser + filters | Objective: plain | Objective: parser + filters | Avg results returned (b) |
|---|---|---|---|---|---|---|
| Both sets | 21 | 27.6% | 92.4% | 55.2% | 92.9% | 9.3 |
| Dev set only | 15 | 33.3% | 99.3% | 62.0% | 100.0% | 10.0 |
| Held-out only | 6 | 13.3% | 75.0% | 38.3% | 75.0% | 7.5 |

Caveats: 21 queries is a small sample, and only 6 are from the held-out set. must-have/exclude are judged by keyword rules over title and features, and the filter uses the same rules, so for those two the figure checks the mechanism, not relevance. Satisfying a constraint does not prove a result is relevant; no relevance labels exist for the filtered lists. Filtering can return fewer than 10 results.

## Cost and latency

Prompt v2 uses 647 prompt and 94 completion tokens per query on average (147 successful calls). At Groq's list price for `openai/gpt-oss-20b` ($0.075 input / $0.30 output per 1M tokens, read from console.groq.com/docs/models on 2026-10-24) that is about **$0.000077 per query** ($0.077 per 1,000). The free tier costs nothing but is limited; the rate-limits page listed 30 requests/min, 8K tokens/min, 1K requests/day and 200K tokens/day for this model, but its text did not say which plan that table describes.

| Run | Parser p50 s | Parser p95 s |
|---|---|---|
| Held-out, prompt v2 | 0.78 | 1.32 |
| Dev, prompt v2 | 0.69 | 1.42 |

Latency is the whole `parse()` call (regex + one Groq request) from this laptop, one query at a time.

## Fallback chain

Run with the full chain (Groq, then OpenRouter) at about one request per second on the dev set with prompt v1: {'groq:openai/gpt-oss-20b': 83, 'openrouter:recursive/flint-v1.0:free': 17}. Groq rejected those 17 requests (the status code was not recorded; most likely the free-tier rate limit). Parser p95 for that run was 7.3 s; in separate single-call tests the OpenRouter free model `recursive/flint-v1.0` took about 4 to 5 s. Both are above the app's 3 s budget, so a busy Groq free tier would break the latency target even though the app keeps working. OpenRouter free models were also unreliable on 2026-10-24: Gemma returned HTTP 429 on every attempt, and `nemotron-3-super-120b` returned an upstream 404.

## Limitations

- The gold sets are small (100 dev, 50 held-out) and were written by Claude (the assistant), not by the project owner; the owner has not reviewed them yet.
- Prompt v2 was written after reading v1's errors on the 100 dev queries; the held-out set exists to check that it generalises, and it did (0.72 to 0.92), but with 50 queries the interval is wide.
- v2 prompt tokens are about double v1, which matters for the free-tier token limit.
- English-only regex: Spanish and German price/rating phrases rely on the LLM.
