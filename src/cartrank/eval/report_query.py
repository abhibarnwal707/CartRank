"""Render results/query_layer.md from results/query_layer.json."""

import json

from cartrank.data.paths import RESULTS

# Groq list price for openai/gpt-oss-20b, read from console.groq.com/docs/models on 2026-10-24.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.075, 0.30
FIELDS = ["route", "price_min", "price_max", "brand", "min_rating", "must_have", "exclude"]


def _t(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def _row(label, s, summary=None):
    o = s["overall"]
    cells = [label, str(o["n"])] + [f"{o[f]:.2f}" for f in FIELDS]
    cells += [f"{o['text_f1']:.2f}", f"{o['constraint_intent_exact']:.2f}"]
    if summary is not None:
        cells.append(f"{summary['llm_success']}/{summary['n_queries']}")
    return cells


def write_report() -> None:
    d = json.loads((RESULTS / "query_layer.json").read_text())
    hdr = ["Run", "n"] + FIELDS + ["text F1", "Constraint-intent exact"]
    g1, g2 = d["parser_gold_v1_groq"], d["parser_gold_v2_groq"]
    h1, h2 = d["parser_heldout_v1_groq"], d["parser_heldout_v2_groq"]
    rows_dev = [
        _row("Regex-only (no LLM)", d["parser_gold_regex_only"]),
        _row("LLM prompt v1", g1["scores"]),
        _row("LLM prompt v2 (tuned on this set)", g2["scores"]),
    ]
    rows_held = [
        _row("Regex-only (no LLM)", d["parser_heldout_regex_only"]),
        _row("LLM prompt v1", h1["scores"]),
        _row("LLM prompt v2", h2["scores"]),
    ]
    runs = [g2, h2]
    n_tok = sum(r["summary"]["llm_success"] for r in runs)
    p_tok = sum(r["summary"]["prompt_tokens_mean"] * r["summary"]["llm_success"] for r in runs)
    c_tok = sum(r["summary"]["completion_tokens_mean"] * r["summary"]["llm_success"] for r in runs)
    p_mean, c_mean = p_tok / n_tok, c_tok / n_tok
    cost = (p_mean * PRICE_IN_PER_M + c_mean * PRICE_OUT_PER_M) / 1e6
    lat = h2["summary"]["latency_s"]
    glat = g2["summary"]["latency_s"]
    chain = d["parser_gold_v1_chain"]["summary"]
    c = d["constraints"]
    cs_rows = []
    for key, label in (
        ("both_sets", "Both sets"),
        ("gold_only", "Dev set only"),
        ("heldout_only", "Held-out only"),
    ):
        v = c[key]
        cs_rows.append(
            [
                label,
                str(v["n"]),
                f"{v['plain']['all']:.1%}",
                f"{v['parsed']['all']:.1%}",
                f"{v['plain']['objective']:.1%}",
                f"{v['parsed']['objective']:.1%}",
                f"{v['parsed']['returned']:.1f}",
            ]
        )
    out = [
        "# Query understanding layer (Phase 5)\n",
        "One LLM call turns a free-text query into a typed `QueryIntent` (route, text, price "
        "range, brand, minimum rating, must-have and exclude terms). A deterministic regex "
        "pre-pass runs first and wins on price and rating. If the LLM fails, the app falls back "
        "to the regex result with route `search`.\n",
        "Model: Groq `openai/gpt-oss-20b` (primary), then OpenRouter free models in order "
        "(`recursive/flint-v1.0:free`, `nvidia/nemotron-3.5-super-vl:free`, "
        "`google/gemma-4-26b-a4b-it:free`). All parser rows below use Groq only, paced under the "
        "free-tier request limit.\n",
        "## Parser accuracy\n",
        "Fields are scored by exact match (brand case-insensitive; must-have/exclude as sets; "
        "hyphens ignored). `text` is mean token F1. *Constraint-intent exact* = every field except "
        "text correct. `category` is not scored (no canonical vocabulary).\n",
        "### Held-out set (50 queries written before any model output on them was seen, run once "
        "per frozen prompt): the number to quote\n",
        _t(
            hdr + ["LLM answered"],
            [
                r + [x]
                for r, x in zip(
                    rows_held,
                    [
                        "-",
                        f"{h1['summary']['llm_success']}/50",
                        f"{h2['summary']['llm_success']}/50",
                    ],
                    strict=True,
                )
            ],
        ),
        "\n### Dev set (100 queries, used while writing the prompts, so v2 is optimistic here)\n",
        _t(
            hdr + ["LLM answered"],
            [
                r + [x]
                for r, x in zip(
                    rows_dev,
                    [
                        "-",
                        f"{g1['summary']['llm_success']}/100",
                        f"{g2['summary']['llm_success']}/100",
                    ],
                    strict=True,
                )
            ],
        ),
        "\nWhen the LLM did not answer (the error was recorded only as `LLMError`; most likely the "
        "free-tier rate limit), the query is scored with its regex-only fallback, so the numbers "
        "include those failures.\n",
        "### Held-out, prompt v2, by query group\n",
        _t(
            ["Group", "n", "Constraint-intent exact", "text F1"],
            [
                [g, str(v["n"]), f"{v['constraint_intent_exact']:.2f}", f"{v['text_f1']:.2f}"]
                for g, v in h2["scores"]["by_group"].items()
            ],
        ),
        "\nGroups have 4 to 6 queries each; read them as indicative only.\n",
        "## Constraint satisfaction, end to end\n",
        "Queries with at least two stated constraints (price, rating, brand, each must-have and "
        "exclude term). Top-10 of (a) plain hybrid search + cross-encoder on the raw query vs (b) "
        "parser + hard filters. Constraints are the gold labels. Cell = share of the 10 result "
        "slots whose product satisfies every constraint (a slot with no result counts as a "
        'failure). "Objective" counts only price, rating and brand (catalog metadata).\n',
        _t(
            [
                "Subset",
                "Queries",
                "All: plain",
                "All: parser + filters",
                "Objective: plain",
                "Objective: parser + filters",
                "Avg results returned (b)",
            ],
            cs_rows,
        ),
        "\nCaveats: 21 queries is a small sample, and only 6 are from the held-out set. "
        "must-have/exclude are judged by keyword rules over title and features, and the filter uses "
        "the same rules, so for those two the figure checks the mechanism, not relevance. "
        "Satisfying a constraint does not prove a result is relevant; no relevance labels exist for "
        "the filtered lists. Filtering can return fewer than 10 results.\n",
        "## Cost and latency\n",
        f"Prompt v2 uses {p_mean:.0f} prompt and {c_mean:.0f} completion tokens per query on "
        f"average ({n_tok} successful calls). At Groq's list price for `openai/gpt-oss-20b` "
        f"(${PRICE_IN_PER_M:.3f} input / ${PRICE_OUT_PER_M:.2f} output per 1M tokens, read from "
        f"console.groq.com/docs/models on 2026-10-24) that is about **${cost:.6f} per query** "
        f"(${cost * 1000:.3f} per 1,000). The free tier costs nothing but is limited; the "
        "rate-limits page listed 30 requests/min, 8K tokens/min, 1K requests/day and 200K "
        "tokens/day for this model, but its text did not say which plan that table describes.\n",
        _t(
            ["Run", "Parser p50 s", "Parser p95 s"],
            [
                ["Held-out, prompt v2", f"{lat['p50']:.2f}", f"{lat['p95']:.2f}"],
                ["Dev, prompt v2", f"{glat['p50']:.2f}", f"{glat['p95']:.2f}"],
            ],
        ),
        "\nLatency is the whole `parse()` call (regex + one Groq request) from this laptop, one "
        "query at a time.\n",
        "## Fallback chain\n",
        f"Run with the full chain (Groq, then OpenRouter) at about one request per second on the "
        f"dev set with prompt v1: {chain['providers']}. Groq rejected those 17 requests (the status "
        f"code was not recorded; most likely the free-tier rate limit). Parser p95 for that run was "
        f"{chain['latency_s']['p95']:.1f} s; in separate single-call tests the OpenRouter free model "
        "`recursive/flint-v1.0` took about 4 to 5 s. Both are above the app's 3 s budget, so a busy "
        "Groq free tier would break the latency target even though the app keeps working. "
        "OpenRouter free models were also unreliable on 2026-10-24: Gemma returned HTTP 429 on every "
        "attempt, and `nemotron-3-super-120b` returned an upstream 404.\n",
        "## Limitations\n",
        "- The gold sets are small (100 dev, 50 held-out) and were written by Claude (the assistant), "
        "not by the project owner; the owner has not reviewed them yet.\n"
        "- Prompt v2 was written after reading v1's errors on the 100 dev queries; the held-out "
        "set exists to check that it generalises, and it did (0.72 to 0.92), but with 50 queries "
        "the interval is wide.\n"
        "- v2 prompt tokens are about double v1, which matters for the free-tier token limit.\n"
        "- English-only regex: Spanish and German price/rating phrases rely on the LLM.\n",
    ]
    (RESULTS / "query_layer.md").write_text("\n".join(out))


if __name__ == "__main__":
    write_report()
