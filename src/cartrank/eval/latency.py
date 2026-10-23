"""Latency and memory of the real serving pipeline on 50 scripted queries.

python -m cartrank.eval.latency <no-llm|llm> [pace_seconds]

Run in a fresh process so cold start is real. Results are labelled with where they were
measured; Streamlit Cloud numbers are measured separately after deployment.
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from cartrank.data.paths import RESULTS, ROOT

QUERIES = ROOT / "evals" / "queries" / "demo_queries.jsonl"


def rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.strip()) / 1024


def main(mode: str, pace: float) -> None:
    if mode not in ("no-llm", "llm"):
        raise SystemExit("usage: python -m cartrank.eval.latency <no-llm|llm> [pace_seconds]")
    if mode == "no-llm":
        os.environ["CARTRANK_NO_LLM"] = "1"
    from cartrank.serve.loader import load_pipeline

    rss_start = rss_mb()
    t0 = time.perf_counter()
    pipe = load_pipeline(device="cpu")
    pipe.ce_batch = int(os.environ.get("CE_BATCH", pipe.ce_batch))
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()
    queries = [json.loads(line)["query"] for line in QUERIES.read_text().splitlines()]

    t0 = time.perf_counter()
    pipe.search(queries[0])  # first query pays lazy cross-encoder load and warm-up
    first_s = time.perf_counter() - t0
    cold_start_s = load_s + first_s

    rows, providers = [], {}
    for q in queries:
        t = time.perf_counter()
        r = pipe.search(q)
        wall = (time.perf_counter() - t) * 1000
        rows.append(
            {
                "query": q,
                "wall_ms": wall,
                "llm_used": r.llm_used,
                "route": r.route_used,
                "n_results": len(r.results),
                "timings": r.timings_ms,
            }
        )
        u = r.debug.get("parser", {}).get("usage") or {}
        if u:
            key = f"{u.get('provider')}:{u.get('model')}"
            providers[key] = providers.get(key, 0) + 1
        if mode == "llm":
            time.sleep(pace)
    wall = np.array([x["wall_ms"] for x in rows])
    stages = sorted({k for x in rows for k in x["timings"] if k != "total"})
    stage_stats = {
        s: {
            "p50": float(np.percentile([x["timings"][s] for x in rows if s in x["timings"]], 50)),
            "p95": float(np.percentile([x["timings"][s] for x in rows if s in x["timings"]], 95)),
        }
        for s in stages
    }
    out = {
        "where": f"local {platform.system()} {platform.machine()}, CPU, Python {platform.python_version()}",
        "mode": mode,
        "pace_seconds": pace if mode == "llm" else 0,
        "n_queries": len(rows),
        "load_seconds": load_s,
        "first_query_seconds": first_s,
        "cold_start_seconds_components": cold_start_s,
        "rss_mb": {"before_load": rss_start, "after_load": rss_loaded, "after_queries": rss_mb()},
        "wall_ms": {
            "p50": float(np.percentile(wall, 50)),
            "p95": float(np.percentile(wall, 95)),
            "max": float(wall.max()),
        },
        "stages_ms": stage_stats,
        "llm_answered": sum(x["llm_used"] for x in rows),
        "providers": providers,
        "regex_fallbacks": sum(not x["llm_used"] for x in rows) if mode == "llm" else None,
        "empty_results": sum(x["n_results"] == 0 for x in rows),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "latency.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    # CE_BATCH set -> a batch-size sweep run, stored under its own key
    key = (
        f"local_{mode}_ce{os.environ['CE_BATCH']}"
        if os.environ.get("CE_BATCH")
        else f"local_{mode}"
    )
    out["ce_batch"] = pipe.ce_batch
    cur[key] = out
    path.write_text(json.dumps(cur, indent=2))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "mode",
                    "load_seconds",
                    "first_query_seconds",
                    "rss_mb",
                    "wall_ms",
                    "llm_answered",
                    "providers",
                    "empty_results",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.5)
