"""Render results/search_dev.json into results/search_baselines.md."""

import json

from cartrank.data.paths import RESULTS

# Main ladder (playbook): (label, {mode: key in results/search_dev.json "ladder"}).
# In rerank mode the cross-encoder orders every judged candidate whatever the first stage is,
# so its row is shared by the base and fine-tuned pipelines.
CE_K = 50  # working point chosen on dev: NDCG vs CPU latency, see the rerank tuning table
ROWS = [
    ("BM25", {"rerank": "bm25", "full": "bm25"}),
    ("Dense, bge-small off-the-shelf", {"rerank": "dense_base", "full": "dense_base"}),
    ("Dense, bge-small fine-tuned on ESCI", {"rerank": "dense_ft", "full": "dense_ft"}),
    ("Hybrid RRF (BM25 + fine-tuned dense)", {"rerank": "hybrid_ft", "full": "hybrid_ft"}),
    (
        f"Hybrid-FT + cross-encoder rerank (full-corpus: top-{CE_K} of hybrid)",
        {"rerank": "hybrid_base_ce", "full": f"hybrid_ft_ce_k{CE_K}"},
    ),
]
# Off-the-shelf hybrid kept for reference (Phase 1 result and its rerank).
REF_ROWS = [
    ("Hybrid RRF (BM25 + off-the-shelf dense)", {"rerank": "hybrid_base", "full": "hybrid_base"}),
    (
        f"Hybrid-base + cross-encoder rerank (full-corpus: top-{CE_K})",
        {"rerank": "hybrid_base_ce", "full": f"hybrid_base_ce_k{CE_K}"},
    ),
]
MODE_TITLES = {
    "rerank": "Rerank mode (standard ESCI Task 1): rank each query's judged products only",
    "full": "Full-corpus retrieval mode: retrieve from all US small-set products",
}
CAVEAT = (
    "Unjudged products count as non-relevant, so full-corpus scores underestimate true quality "
    "and are not comparable to rerank-mode scores."
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _ladder(data: dict, mode: str, spec=None) -> str:
    rows, prev = [], None
    for label, keys in spec or ROWS:
        m = data["ladder"].get(f"{keys[mode]}|{mode}")
        if m is None:
            continue
        delta = "-" if prev is None else f"{m['ndcg@10'] - prev:+.4f}"
        rows.append(
            [label, f"{m['ndcg@10']:.4f}", delta, f"{m['recall@100']:.4f}", f"{m['mrr@10']:.4f}"]
        )
        prev = m["ndcg@10"]
    return _table(["System", "NDCG@10", "Δ vs previous row", "Recall@100", "MRR@10"], rows)


def write_report() -> None:
    data = json.loads((RESULTS / "search_dev.json").read_text())
    n = next(iter(data["ladder"].values()))["n_queries"]
    out = [
        "# Search baselines (ESCI dev)\n",
        f"Dev split: {n:,} queries (10% of train query_ids, split by query). "
        "Test split not used. NDCG@10 uses graded gains E=1.0, S=0.1, C=0.01, I=0; "
        "Recall@100 counts Exact as relevant (queries with no Exact excluded); "
        "MRR@10 is the first Exact result.\n",
    ]
    for mode, title in MODE_TITLES.items():
        out.append(f"## {title}\n")
        out.append(_ladder(data, mode))
        if mode == "rerank":
            out.append(
                "\nRecall@100 is trivially 1.0 here: each query has about 20 judged products, so "
                "all of them fit in the top 100. Hybrid RRF parameters were tuned on full-corpus "
                "NDCG only, so hybrid can trail the best single retriever in this mode. The "
                "cross-encoder row is the same for any first stage here: it orders all judged "
                "candidates itself."
            )
        if mode == "full":
            out.append(f"\n{CAVEAT}")
        out.append("")

    out.append("## Reference: off-the-shelf hybrid (Phase 1) and its rerank\n")
    for mode in MODE_TITLES:
        out.append(f"{mode} mode:\n")
        out.append(_ladder(data, mode, REF_ROWS))
        out.append("")

    if "hybrid_best_ft" in data:
        b = data["hybrid_best_ft"]
        out.append(
            f"## Hybrid tuning, fine-tuned encoder (dev, full-corpus NDCG@10)\n\n"
            f"Best: k={b['k']}, depth={b['depth']} (NDCG@10 {b['ndcg@10']:.4f}).\n"
        )
        out.append(
            _table(
                ["k", "depth", "NDCG@10", "Recall@100"],
                [
                    [str(r["k"]), str(r["depth"]), f"{r['ndcg@10']:.4f}", f"{r['recall@100']:.4f}"]
                    for r in data["hybrid_tuning_ft"]
                ],
            )
        )
        out.append("")

    if "hybrid_best" in data:
        b = data["hybrid_best"]
        out.append(
            f"## Hybrid tuning (dev, full-corpus NDCG@10)\n\nBest: k={b['k']}, depth={b['depth']} "
            f"(NDCG@10 {b['ndcg@10']:.4f}).\n"
        )
        out.append(
            _table(
                ["k", "depth", "NDCG@10", "Recall@100"],
                [
                    [str(r["k"]), str(r["depth"]), f"{r['ndcg@10']:.4f}", f"{r['recall@100']:.4f}"]
                    for r in data["hybrid_tuning"]
                ],
            )
        )
        out.append("")

    if "rerank_tuning_base" in data:
        lat = data["rerank_latency_ms_base"]
        mdl = data["rerank_model"]
        out.append(
            f"## Cross-encoder rerank (full-corpus, hybrid first stage)\n\n"
            f"Model `{mdl['name']}` ({mdl['params'] / 1e6:.1f}M params), input = query + product "
            f"text truncated to {mdl['max_length']} tokens, scored on CPU. In rerank mode the "
            "cross-encoder orders every judged candidate, so k does not apply. Latency is the "
            "cross-encoder stage alone, 50 dev queries, local MacBook Air CPU.\n"
        )
        base = data["ladder"]["hybrid_base|full"]["ndcg@10"]
        out.append(
            _table(
                ["k", "NDCG@10", "Δ vs hybrid", "Recall@100", "MRR@10", "p50 ms", "p95 ms"],
                [
                    [
                        str(r["k"]),
                        f"{r['ndcg@10']:.4f}",
                        f"{r['ndcg@10'] - base:+.4f}",
                        f"{r['recall@100']:.4f}",
                        f"{r['mrr@10']:.4f}",
                        f"{lat[f'k={r["k"]}']['p50_ms']:.0f}",
                        f"{lat[f'k={r["k"]}']['p95_ms']:.0f}",
                    ]
                    for r in data["rerank_tuning_base"]
                ],
            )
        )
        out.append(
            f"\nWorking point k={CE_K}: most of the gain of k=100 at about half its latency "
            "(chosen on dev, not test).\n"
        )

    if "latency_ms" in data:
        lat, idx = data["latency_ms"], data["index"]
        out.append(
            "## Latency and index size\n\nLatency is one full-corpus query at a time on the "
            "local MacBook Air (query encoding on MPS for dense), 200 dev queries, top-100, "
            f"{idx['corpus_size']:,} products.\n"
        )
        out.append(
            _table(
                ["System", "p50 ms", "p95 ms"],
                [[k, f"{v['p50_ms']:.1f}", f"{v['p95_ms']:.1f}"] for k, v in lat.items()],
            )
        )
        out.append(
            f"\nBM25 index: build {idx['bm25_build_seconds']:.0f} s, {idx['bm25_index_mb']:.0f} MB "
            f"on disk. Dense vectors: {idx['dense_dim']}-dim float32, "
            f"{idx['dense_vectors_mb']:.0f} MB (FAISS flat inner product).\n"
        )

    if "ablation_text_bm25" in data:
        out.append("## Ablation: product text (BM25)\n")
        labels = {
            "title": "title",
            "tbc": "title + brand + color",
            "tbcb": "title + brand + color + bullets (default)",
            "tbcbd": "title + brand + color + bullets + description",
        }
        rows = [
            [
                labels[v],
                f"{r['rerank']['ndcg@10']:.4f}",
                f"{r['full']['ndcg@10']:.4f}",
                f"{r['full']['recall@100']:.4f}",
            ]
            for v, r in data["ablation_text_bm25"].items()
        ]
        out.append(_table(["Text", "NDCG@10 rerank", "NDCG@10 full", "Recall@100 full"], rows))
        out.append("")

    if "ablation_text_dense" in data:
        out.append(
            "## Ablation: product text and query prefix (dense, bge-small)\n\n"
            "Rerank mode only (the corpus is the dev queries' judged products).\n"
        )
        rows = [
            [k.split("|")[0], k.split("|")[1], f"{r['ndcg@10']:.4f}", f"{r['mrr@10']:.4f}"]
            for k, r in data["ablation_text_dense"].items()
        ]
        out.append(_table(["Text variant", "Query prefix", "NDCG@10", "MRR@10"], rows))
        out.append("")

    (RESULTS / "search_baselines.md").write_text("\n".join(out))
