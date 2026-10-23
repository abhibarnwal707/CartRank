"""Phase 5 driver.

python -m cartrank.eval.run_phase5 parser <groq|chain> <v1|v2> <gold|heldout>
`gold` = 100 queries used while developing the prompt (dev); `heldout` = 50 queries written

`gold` = 100 queries used while developing the prompt (dev); `heldout` = 50 queries written
before any model output on them was seen, run once per frozen prompt. Results land in
results/query_layer.json under parser_<dataset>_<prompt>_<mode>.
"""

import json
import sys
import time

import numpy as np

from cartrank import config
from cartrank.data.paths import RESULTS, ROOT
from cartrank.eval.parser_eval import score
from cartrank.eval.search_eval import merge_results
from cartrank.query import parser
from cartrank.query.llm_clients import FallbackClient, default_client, groq
from cartrank.query.schema import QueryIntent

DATASETS = {
    "gold": ROOT / "evals" / "queries" / "parser_gold.jsonl",
    "heldout": ROOT / "evals" / "queries" / "parser_heldout.jsonl",
}
JSON = "query_layer.json"


def load_set(name: str) -> list[dict]:
    rows = [json.loads(line) for line in DATASETS[name].read_text().splitlines()]
    for r in rows:
        r["gold"] = QueryIntent.model_validate(r.pop("expected"))
    return rows


def _summary(records: list[dict], n: int, mode: str) -> dict:
    ok = [x for x in records if x["used_llm"]]
    lat = np.array([x["latency_s"] for x in ok])
    providers: dict[str, int] = {}
    for x in ok:
        key = f"{x['usage'].get('provider')}:{x['usage'].get('model')}"
        providers[key] = providers.get(key, 0) + 1
    return {
        "n_queries": n,
        "llm_success": len(ok),
        "llm_failures": [
            {"id": x["id"], "error": x["error"]} for x in records if not x["used_llm"]
        ],
        "providers": providers,
        "latency_s": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "prompt_tokens_mean": float(np.mean([x["usage"]["prompt_tokens"] for x in ok])),
        "completion_tokens_mean": float(np.mean([x["usage"]["completion_tokens"] for x in ok])),
        "mode": mode,
    }


def run_parser_eval(mode: str, prompt: str, dataset: str) -> None:
    """mode "chain": Groq then OpenRouter fallbacks (what the app does). mode "groq": the primary
    model only, paced under the free-tier request limit, so the numbers describe one model."""
    gold = load_set(dataset)
    config.load_env()
    client = default_client() if mode == "chain" else FallbackClient([groq()])
    pace = 0.3 if mode == "chain" else 3.0  # free tier: 30 requests/min and 8K tokens/min
    regex_rows, llm_rows, records = [], [], []
    for r in gold:
        base = parser.parse(r["query"], None)
        regex_rows.append({"group": r["group"], "gold": r["gold"], "pred": base.intent})
        res = parser.parse(r["query"], client, system=parser.PROMPTS[prompt])
        llm_rows.append({"group": r["group"], "gold": r["gold"], "pred": res.intent})
        records.append(
            {
                "id": r["id"],
                "query": r["query"],
                "group": r["group"],
                "used_llm": res.used_llm,
                "latency_s": res.latency_s,
                "error": res.error,
                "usage": res.usage,
                "pred": res.intent.model_dump(),
            }
        )
        time.sleep(pace)
    key = f"parser_{dataset}_{prompt}_{mode}"
    merge_results(
        JSON,
        {
            f"parser_{dataset}_regex_only": score(regex_rows),
            key: {
                "scores": score(llm_rows),
                "summary": _summary(records, len(gold), mode),
                "records": records,
                "prompt": prompt,
                "model": config.GROQ_MODEL,
            },
        },
    )
    print(key, json.dumps(_summary(records, len(gold), mode)["providers"]))
    print("regex-only", json.dumps(score(regex_rows)["overall"]))
    print("llm       ", json.dumps(score(llm_rows)["overall"]))


def run_constraints(prompt: str = "v2") -> None:
    """Plain hybrid+CE search vs parser + hard filters, on queries with >= 2 stated constraints.

    Uses the stored parser predictions (no new API calls). Constraints are the GOLD labels.
    """
    from cartrank.eval.constraints_eval import OBJECTIVE, n_constraints, satisfies_all
    from cartrank.search.catalog_index import CatalogSearch, hybrid_setting, search_intent

    d = json.loads((RESULTS / JSON).read_text())
    k, depth = hybrid_setting("ft")
    cs = CatalogSearch("ft", "artifacts/models/bge-small-esci-ft", k, depth)
    rows = []
    for dataset in ("gold", "heldout"):
        run = d[f"parser_{dataset}_{prompt}_groq"]
        preds = {x["id"]: QueryIntent.model_validate(x["pred"]) for x in run["records"]}
        for r in load_set(dataset):
            if n_constraints(r["gold"]) < 2:
                continue
            plain = cs.final(r["query"])[:10]
            parsed = search_intent(cs, preds[r["id"]], r["query"])[:10]
            g = r["gold"]
            has_obj = n_constraints(g, OBJECTIVE) > 0
            rows.append(
                {
                    "dataset": dataset,
                    "id": r["id"],
                    "query": r["query"],
                    "group": r["group"],
                    "has_objective": has_obj,
                    "plain": {
                        "all": sum(satisfies_all(cs.item(a), g) for a in plain) / 10,
                        "objective": sum(satisfies_all(cs.item(a), g, OBJECTIVE) for a in plain)
                        / 10,
                        "returned": len(plain),
                    },
                    "parsed": {
                        "all": sum(satisfies_all(cs.item(a), g) for a in parsed) / 10,
                        "objective": sum(satisfies_all(cs.item(a), g, OBJECTIVE) for a in parsed)
                        / 10,
                        "returned": len(parsed),
                    },
                }
            )

    def agg(sub: list[dict]) -> dict:
        obj = [x for x in sub if x["has_objective"]]
        out = {"n": len(sub), "n_with_objective": len(obj)}
        for sys_ in ("plain", "parsed"):
            out[sys_] = {
                "all": float(np.mean([x[sys_]["all"] for x in sub])),
                "objective": float(np.mean([x[sys_]["objective"] for x in obj])) if obj else None,
                "returned": float(np.mean([x[sys_]["returned"] for x in sub])),
            }
        return out

    result = {
        "prompt": prompt,
        "both_sets": agg(rows),
        "heldout_only": agg([x for x in rows if x["dataset"] == "heldout"]),
        "gold_only": agg([x for x in rows if x["dataset"] == "gold"]),
        "per_query": rows,
    }
    merge_results(JSON, {"constraints": result})
    for key in ("both_sets", "heldout_only", "gold_only"):
        print(key, json.dumps(result[key]))


def rescore() -> None:
    """Re-score stored predictions with the current scorer (e.g. after hyphen normalisation)."""
    path = RESULTS / JSON
    d = json.loads(path.read_text())
    for key, run in d.items():
        if not (key.startswith("parser_") and isinstance(run, dict) and "records" in run):
            continue
        dataset = key.split("_")[1]
        gold = {r["id"]: r for r in load_set(dataset)}
        rows = [
            {
                "group": x["group"],
                "gold": gold[x["id"]]["gold"],
                "pred": QueryIntent.model_validate(x["pred"]),
            }
            for x in run["records"]
        ]
        run["scores"] = score(rows)
        print("rescored", key, run["scores"]["overall"]["constraint_intent_exact"])
    path.write_text(json.dumps(d, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "parser":
        run_parser_eval(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "constraints":
        run_constraints()
    else:
        rescore()
