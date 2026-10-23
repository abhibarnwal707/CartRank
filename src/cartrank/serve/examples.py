"""Precomputed example responses, so the demo opens full without spending any LLM tokens.

python -m cartrank.serve.examples   regenerates app/examples.json from the real pipeline.
"""

import dataclasses
import json

from cartrank.data.paths import ROOT
from cartrank.query.schema import QueryIntent
from cartrank.serve.pipeline import Response, Result

PATH = ROOT / "app" / "examples.json"
QUERIES = [
    "black gel pens without clips under $10",
    "swingline stapler 4 stars and up",
    "notebook not spiral bound between $5 and $15",
    "something like the swingline 747 stapler",
]


def to_dict(r: Response) -> dict:
    """Serialisable part of a Response (timings are deliberately left out)."""
    return {
        "query": r.query,
        "intent": r.intent.model_dump(),
        "route_used": r.route_used,
        "notes": r.notes,
        "llm_used": r.llm_used,
        "debug_candidates": r.debug.get("candidates", {}),
        "results": [dataclasses.asdict(x) for x in r.results],
        "reference": dataclasses.asdict(r.reference) if r.reference else None,
    }


def from_dict(d: dict) -> Response:
    return Response(
        query=d["query"],
        intent=QueryIntent.model_validate(d["intent"]),
        results=[Result(**x) for x in d["results"]],
        route_used=d["route_used"],
        notes=list(d["notes"]),
        debug={"candidates": d["debug_candidates"]},
        llm_used=d["llm_used"],
        reference=Result(**d["reference"]) if d.get("reference") else None,
    )


def load() -> list[Response]:
    return [from_dict(d) for d in json.loads(PATH.read_text())] if PATH.exists() else []


def build() -> None:
    from cartrank.serve.loader import load_pipeline

    pipe = load_pipeline()
    out = []
    for q in QUERIES:
        r = pipe.search(q)
        print(f"{q!r}: route={r.route_used} llm={r.llm_used} results={len(r.results)}")
        if not r.llm_used:
            raise SystemExit("LLM parsing was unavailable; refusing to save regex-only examples")
        out.append(to_dict(r))
    PATH.write_text(json.dumps(out, indent=1))
    print("wrote", PATH)


if __name__ == "__main__":
    build()
