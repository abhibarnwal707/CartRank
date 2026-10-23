"""Serving pipeline: parse -> route -> (hybrid retrieve -> hard filter -> rerank) or neighbours.

Explanations are templated from catalog fields and the parsed constraints; nothing here lets an
LLM write claims about a product.
"""

import time
from dataclasses import dataclass, field

from cartrank.eval.constraints_eval import _tokens, satisfies_all
from cartrank.query import parser
from cartrank.query.schema import QueryIntent
from cartrank.search.hybrid import rrf
from cartrank.search.rerank import load_cross_encoder

STOP = {"the", "a", "an", "for", "of", "and", "or", "with", "without", "to", "in", "on", "my"}


@dataclass
class Result:
    asin: str
    title: str
    store: str | None
    price: float
    rating: float
    rating_number: int
    image_url: str | None
    why: str


@dataclass
class Response:
    query: str
    intent: QueryIntent
    results: list[Result]
    route_used: str
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    llm_used: bool = False
    reference: Result | None = None


def matched_terms(text: str, item: dict, limit: int = 4) -> list[str]:
    blob = set(_tokens(f"{item['title']} {' '.join(item['features'])}"))
    out = []
    for t in _tokens(text):
        if t not in STOP and t in blob and t not in out:
            out.append(t)
    return out[:limit]


def constraint_notes(item: dict, c: QueryIntent) -> list[str]:
    notes = []
    if c.price_min is not None or c.price_max is not None:
        lo = f"${c.price_min:g}" if c.price_min is not None else None
        hi = f"${c.price_max:g}" if c.price_max is not None else None
        rng = f"{lo} to {hi}" if lo and hi else f"at least {lo}" if lo else f"at most {hi}"
        notes.append(f"price ${item['price']:.2f} ({rng})")
    if c.min_rating is not None:
        notes.append(f"rated {item['average_rating']:.1f} (min {c.min_rating:g})")
    if c.brand:
        notes.append(f"brand matches '{c.brand}'")
    for m in c.must_have:
        notes.append(f"mentions '{m}'")
    for e in c.exclude:
        notes.append(f"no '{e}' mentioned")
    return notes


def why_search(item: dict, text: str, c: QueryIntent, applied: bool) -> str:
    bits = []
    terms = matched_terms(text, item)
    bits.append("matches " + ", ".join(f"'{t}'" for t in terms) if terms else "semantic match")
    if applied:
        bits += constraint_notes(item, c)
    return "; ".join(bits)


def _has_constraints(c: QueryIntent) -> bool:
    return any(
        [
            c.price_min is not None,
            c.price_max is not None,
            c.min_rating is not None,
            c.brand,
            c.must_have,
            c.exclude,
        ]
    )


class Pipeline:
    def __init__(
        self, cs, tt, client, k: int = 10, ce_k: int = 50, pool: int = 300, ce_batch: int = 4
    ):
        self.cs, self.tt, self.client = cs, tt, client
        self.ce_batch = ce_batch
        self.k, self.ce_k, self.pool = k, ce_k, pool
        self.catalog_ids = set(cs.ids)
        self._ce = None

    def _result(self, asin: str, why: str) -> Result:
        row = self.cs.item(asin)
        meta = self.cs.meta(asin)
        return Result(
            asin,
            row["title"],
            row["store"],
            row["price"],
            row["average_rating"],
            int(meta["rating_number"]),
            meta["image_url"],
            why,
        )

    def _rerank(self, q: str, asins: list[str]) -> list[str]:
        if not asins:
            return []
        if self._ce is None:
            self._ce = self.cs.ce or load_cross_encoder()
            self.cs.ce = self._ce
        scores = self._ce.predict([(q, self.cs.text[a]) for a in asins], batch_size=self.ce_batch)
        return [a for a, _ in sorted(zip(asins, scores, strict=True), key=lambda x: -x[1])]

    def lookup(self, text: str, n: int = 6) -> list[Result]:
        """Cheap retrieval for the item picker: hybrid top-n, no LLM, no reranker."""
        bm = self.cs.bm25.search(text, self.cs.depth)
        dn = self.cs.dense(text, self.cs.depth)
        top = rrf([bm, dn], k=self.cs.k, depth=self.cs.depth)[:n]
        return [self._result(a, "") for a in top]

    # ------------------------------------------------------------------ search
    def search(self, query: str) -> Response:
        t0 = time.perf_counter()
        pr = parser.parse(query, self.client)
        intent, timings = pr.intent, {"parse": (time.perf_counter() - t0) * 1000}
        debug = {"parser": {"llm_used": pr.used_llm, "error": pr.error, "usage": pr.usage}}
        if intent.route == "similar_to" and intent.text:
            resp = self._similar(query, intent, timings, debug)
        else:
            resp = self._search(query, intent, timings, debug)
        resp.llm_used = pr.used_llm
        if not pr.used_llm:
            why = (
                "no LLM keys configured" if pr.error == "no_client" else "the language model failed"
            )
            resp.notes.append(
                f"Regex-only mode ({why}): only price and rating phrases were parsed."
            )
        return resp

    def _search(self, query, intent, timings, debug) -> Response:
        q = intent.text or query
        depth = self.cs.depth
        t = time.perf_counter()
        bm = self.cs.bm25.search(q, depth)
        timings["bm25"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        dn = self.cs.dense(q, depth)
        timings["dense"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        fused = rrf([bm, dn], k=self.cs.k, depth=depth)[: self.pool]
        timings["fuse"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        notes, applied = [], _has_constraints(intent)
        kept = [a for a in fused if satisfies_all(self.cs.item(a), intent)] if applied else fused
        if applied and not kept:
            kept, applied = fused, False
            notes.append("No product satisfied every constraint, so the filters were relaxed.")
        timings["filter"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        head = kept[: self.ce_k]
        ranked = self._rerank(q, head) + kept[self.ce_k :]
        timings["rerank"] = (time.perf_counter() - t) * 1000

        top = ranked[: self.k]
        if intent.route == "recommend":
            notes.append(
                "Recommendations need a history: pick items you like in the sidebar for "
                "personalised suggestions. Showing search results for your text meanwhile."
            )
        results = [self._result(a, why_search(self.cs.item(a), q, intent, applied)) for a in top]
        debug["candidates"] = {
            "bm25": len(bm),
            "dense": len(dn),
            "fused": len(fused),
            "after_filter": len(kept),
            "reranked": len(head),
            "bm25_top3": [self.cs.item(a)["title"][:70] for a in bm[:3]],
            "dense_top3": [self.cs.item(a)["title"][:70] for a in dn[:3]],
        }
        timings["total"] = sum(timings.values())
        return Response(query, intent, results, "search", notes, timings, debug)

    # -------------------------------------------------------------- similar_to
    def _similar(self, query, intent, timings, debug) -> Response:
        t = time.perf_counter()
        bm = self.cs.bm25.search(intent.text, self.cs.depth)
        dn = self.cs.dense(intent.text, self.cs.depth)
        cands = rrf([bm, dn], k=self.cs.k, depth=self.cs.depth)[:10]
        ref = self._rerank(intent.text, cands)[0]
        timings["find_reference"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        near = self.tt.similar(ref, k=self.pool, allowed=self.catalog_ids)
        applied = _has_constraints(intent)
        notes = []
        asins = [a for a, _ in near if not applied or satisfies_all(self.cs.item(a), intent)]
        if applied and not asins:
            asins, applied = [a for a, _ in near], False
            notes.append(
                "No similar product satisfied every constraint, so the filters were relaxed."
            )
        timings["neighbours"] = (time.perf_counter() - t) * 1000
        ref_title = self.cs.item(ref)["title"]
        results = [
            self._result(
                a,
                f"similar to '{ref_title[:60]}' in the learned item space"
                + ("; " + "; ".join(constraint_notes(self.cs.item(a), intent)) if applied else ""),
            )
            for a in asins[: self.k]
        ]
        debug["candidates"] = {
            "reference": ref_title,
            "neighbours_pool": len(near),
            "after_filter": len(asins),
        }
        timings["total"] = sum(timings.values())
        return Response(
            query,
            intent,
            results,
            "similar_to",
            notes,
            timings,
            debug,
            reference=self._result(ref, "reference product"),
        )

    # --------------------------------------------------------------- recommend
    def recommend(self, liked: list[str]) -> Response:
        t = time.perf_counter()
        near = self.tt.recommend(liked, k=self.k, allowed=self.catalog_ids)
        timings = {"user_tower+neighbours": (time.perf_counter() - t) * 1000}
        liked_vecs = {a: self.tt.vecs[self.tt.index[a]] for a in liked if a in self.tt.index}
        results = []
        for a, _ in near:
            v = self.tt.vecs[self.tt.index[a]]
            best = max(liked_vecs, key=lambda x: float(liked_vecs[x] @ v))
            title = self.cs.item(best)["title"][:60]
            results.append(self._result(a, f"because you liked '{title}'"))
        timings["total"] = timings["user_tower+neighbours"]
        return Response(
            "(picked items)",
            QueryIntent(route="recommend"),
            results,
            "recommend",
            timings_ms=timings,
            debug={"liked": liked},
        )
