"""Render results/latency.md from results/latency.json."""

import json

from cartrank.data.paths import RESULTS

BUDGET = {"rss_mb": 2000, "cold_start_s": 90, "p95_ms": 3000}


def _t(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def write_report() -> None:
    d = json.loads((RESULTS / "latency.json").read_text())
    runs = [(k, v) for k, v in d.items() if k in ("local_no-llm", "local_llm")]
    rows = []
    for _, r in runs:
        w, m = r["wall_ms"], r["rss_mb"]
        rows.append(
            [
                "LLM parser on" if r["mode"] == "llm" else "LLM off (regex-only)",
                f"{r['load_seconds']:.1f} + {r['first_query_seconds']:.1f}",
                f"{m['after_load']:,.0f}",
                f"{m['after_queries']:,.0f}",
                f"{w['p50']:,.0f}",
                f"{w['p95']:,.0f}",
                f"{w['max']:,.0f}",
                f"{r['llm_answered']}/{r['n_queries']}",
            ]
        )
    out = [
        "# Serving latency and memory\n",
        f"**Measured locally** on {runs[0][1]['where']}; **not yet measured on Streamlit Cloud** "
        "(the deployed numbers go in a separate section once the app is live). 50 scripted queries "
        "(`evals/queries/demo_queries.jsonl`) through the real pipeline in a fresh process: "
        "parse, hybrid retrieve, hard filter, cross-encoder rerank of the top 50, templated "
        "explanation. Encoder and cross-encoder run on CPU.\n",
        f"Budgets (playbook): RSS <= {BUDGET['rss_mb']:,} MB, cold start <= {BUDGET['cold_start_s']} s, "
        f"p95 <= {BUDGET['p95_ms'] / 1000:g} s per query.\n",
        _t(
            [
                "Mode",
                "Cold start s (load + first query)",
                "RSS after load MB",
                "RSS after 50 queries MB",
                "p50 ms",
                "p95 ms",
                "max ms",
                "LLM answered",
            ],
            rows,
        ),
        "\nCold start here is component load plus the first query (which loads the cross-encoder "
        "lazily). It excludes downloading artifacts from the Hub and starting Streamlit, which "
        "the deployed measurement will add.\n",
    ]
    for _, r in runs:
        if r["mode"] == "llm":
            out.append(
                f"LLM run: one query every {r['pace_seconds']:g} s (a burst would exceed Groq's "
                f"free-tier token limit); providers used {r['providers']}; "
                f"{r['regex_fallbacks']} queries fell back to regex-only parsing.\n"
            )
    out.append("## Per-stage latency (ms)\n")
    for _, r in runs:
        out.append(f"{'LLM on' if r['mode'] == 'llm' else 'LLM off'}:\n")
        out.append(
            _t(
                ["Stage", "p50", "p95"],
                [[s, f"{v['p50']:.1f}", f"{v['p95']:.1f}"] for s, v in r["stages_ms"].items()],
            )
        )
        out.append("")
    sweep = sorted((int(k.rsplit("ce", 1)[1]), v) for k, v in d.items() if "_ce" in k)
    if sweep:
        out.append("## Cross-encoder batch size vs memory (LLM off, 50 queries)\n")
        out.append(
            "Batch 64 made resident memory grow well past the budget while queries ran; "
            "smaller batches fix that without changing the ranking (the top-10 order was "
            "identical on 25 of 25 queries between batch 64 and batch 4). The app uses "
            "batch 4.\n"
        )
        out.append(
            _t(
                ["CE batch", "RSS after load MB", "RSS after 50 queries MB", "p50 ms", "p95 ms"],
                [
                    [
                        str(b),
                        f"{v['rss_mb']['after_load']:,.0f}",
                        f"{v['rss_mb']['after_queries']:,.0f}",
                        f"{v['wall_ms']['p50']:.0f}",
                        f"{v['wall_ms']['p95']:.0f}",
                    ]
                    for b, v in sweep
                ],
            )
        )
        out.append("")
    (RESULTS / "latency.md").write_text("\n".join(out))


if __name__ == "__main__":
    write_report()
