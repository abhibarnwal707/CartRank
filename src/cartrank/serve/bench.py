"""In-app benchmark: run scripted queries through the real pipeline and summarise the timings."""

import time

import numpy as np

from cartrank.serve.sysinfo import info, rss_mb


def run_benchmark(pipe, queries: list[str]) -> dict:
    """`pipe` should have its LLM client disabled so the run spends no API tokens."""
    rss_before = rss_mb()
    rows = []
    for q in queries:
        t = time.perf_counter()
        r = pipe.search(q)
        rows.append(
            {
                "wall_ms": (time.perf_counter() - t) * 1000,
                "timings": r.timings_ms,
                "n_results": len(r.results),
            }
        )
    wall = np.array([x["wall_ms"] for x in rows])
    stages = sorted({k for x in rows for k in x["timings"] if k != "total"})
    return {
        "n_queries": len(rows),
        "llm": "off",
        "server": info(),
        "rss_mb": {"before": round(rss_before), "after": round(rss_mb())},
        "wall_ms": {
            "p50": round(float(np.percentile(wall, 50)), 1),
            "p95": round(float(np.percentile(wall, 95)), 1),
            "max": round(float(wall.max()), 1),
        },
        "stages_ms_p50_p95": {
            s: [
                round(
                    float(np.percentile([x["timings"][s] for x in rows if s in x["timings"]], p)), 1
                )
                for p in (50, 95)
            ]
            for s in stages
        },
        "empty_results": sum(x["n_results"] == 0 for x in rows),
    }
