"""Render results/search_dev.json into results/search_baselines.md."""

import json

from cartrank.data.paths import RESULTS

# Main ladder (playbook): (label, {mode: key in results/search_dev.json "ladder"}).
# In rerank mode the cross-encoder orders every judged candidate whatever the first stage is,
# so its row is shared by the base and fine-tuned pipelines.
CE_K = 50  # working point chosen on dev, see the rerank tuning table
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


def _finetune_ablation(data: dict) -> str:
    """Run 1 vs run 2 of the Kaggle fine-tune. Two settings differ, so this is not a clean test."""
    lad = data["ladder"]
    runs = [
        ("Run 1: 1 epoch, (query, Exact) pairs, 163,752 examples", "ft"),
        ("Run 2: 2 epochs, (query, Exact, Irrelevant) triplets, 95,703 examples", "ft2"),
    ]
    rows = []
    for label, t in runs:
        rows.append(
            [
                label,
                f"{lad[f'dense_{t}|rerank']['ndcg@10']:.4f}",
                f"{lad[f'dense_{t}|full']['ndcg@10']:.4f}",
                f"{lad[f'hybrid_{t}|full']['ndcg@10']:.4f}",
                f"{lad[f'hybrid_{t}_ce_k{CE_K}|full']['ndcg@10']:.4f}",
            ]
        )
    return (
        "## Ablation: fine-tune configuration (dev)\n\n"
        "Run 2 changed two things at once (epochs and hard-negative triplets, which also shrinks "
        "the training set to queries that have an Irrelevant product), so a difference cannot be "
        "attributed to either change alone. Both runs: bge-small, lr 5e-5, batch 128 per GPU on "
        "two T4s.\n\n"
        + _table(
            [
                "Configuration",
                "Dense NDCG@10 (rerank)",
                "Dense NDCG@10 (full)",
                "Hybrid NDCG@10 (full)",
                f"Hybrid + CE top-{CE_K} NDCG@10 (full)",
            ],
            rows,
        )
        + "\n"
    )


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

    if "dense_ft2|full" in data["ladder"]:
        out.append(_finetune_ablation(data))

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
        mdl = data["rerank_model"]
        out.append(
            f"## Cross-encoder rerank (full-corpus, hybrid first stage)\n\n"
            f"Model `{mdl['name']}` ({mdl['params'] / 1e6:.1f}M params), input = query + product "
            f"text truncated to {mdl['max_length']} tokens, scored on CPU. In rerank mode the "
            "cross-encoder orders every judged candidate, so k does not apply.\n"
        )
        base = data["ladder"]["hybrid_base|full"]["ndcg@10"]
        out.append(
            _table(
                ["k", "NDCG@10", "Δ vs hybrid", "Recall@100", "MRR@10"],
                [
                    [
                        str(r["k"]),
                        f"{r['ndcg@10']:.4f}",
                        f"{r['ndcg@10'] - base:+.4f}",
                        f"{r['recall@100']:.4f}",
                        f"{r['mrr@10']:.4f}",
                    ]
                    for r in data["rerank_tuning_base"]
                ],
            )
        )
        by_k = {r["k"]: r["ndcg@10"] for r in data["rerank_tuning_base"]}
        out.append(
            f"\nWorking point k={CE_K} (chosen on dev, not test): k=100 adds only "
            f"{by_k[100] - by_k[CE_K]:+.4f} NDCG@10 over k={CE_K}.\n"
        )

    if "index" in data:
        idx = data["index"]
        out.append(
            f"## Index size\n\n{idx['corpus_size']:,} products. BM25 index: build "
            f"{idx['bm25_build_seconds']:.0f} s, {idx['bm25_index_mb']:.0f} MB on disk. "
            "Dense vectors: "
            f"{idx['dense_dim']}-dim float32, {idx['dense_vectors_mb']:.0f} MB (FAISS flat inner "
            "product).\n"
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
