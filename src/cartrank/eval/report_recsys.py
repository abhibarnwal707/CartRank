"""Render results/recsys.md from results/recsys_dev.json and the Kaggle two-tower results."""

import json
import shutil

from cartrank.data.paths import RESULTS, ROOT

TT_SRC = ROOT / "artifacts" / "two_tower" / "two_tower_results.json"
TT_DST = RESULTS / "two_tower_dev.json"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _row(label: str, m: dict, cold: dict | None, train_s: float, note: str, prev: float | None):
    delta = "-" if prev is None else f"{m['ndcg@10'] - prev:+.4f}"
    return [
        label,
        f"{m['hr@10']:.4f}",
        f"{m['ndcg@10']:.4f}",
        delta,
        f"{cold['hr@10']:.4f}" if cold else "-",
        f"{cold['ndcg@10']:.4f}" if cold else "-",
        f"{train_s:.0f}",
        note,
    ]


def write_report() -> None:
    d = json.loads((RESULTS / "recsys_dev.json").read_text())
    if TT_SRC.exists():
        shutil.copy(TT_SRC, TT_DST)
    tt = json.loads(TT_DST.read_text()) if TT_DST.exists() else {}
    info = d["data"]
    rows, prev = [], None

    def add(label, res, train_s, note):
        nonlocal prev
        rows.append(_row(label, res["all"], res["cold"], train_s, note, prev))
        prev = res["all"]["ndcg@10"]

    add(
        "Popularity",
        d["rows"]["popularity"],
        d["rows"]["popularity"]["train_seconds"],
        "train counts",
    )
    k = d["rows"]["itemknn"]
    add("ItemKNN", k, k["train_seconds"], f"cosine, top-{k['k']} neighbours")
    if "bpr" in d["rows"]:
        b = d["rows"]["bpr"]
        add(
            "BPR-MF (PyTorch)",
            b,
            b["train_seconds"],
            f"dim {b['dim']}, lr {b['lr']}, reg {b['reg']}, epoch {b['best_epoch']}",
        )
    best_tt = {}
    for variant, label in (
        ("id_only", "Two-tower, ID-only item tower"),
        ("id_text", "Two-tower, ID + text item tower"),
    ):
        runs = {n: r for n, r in tt.items() if r["variant"] == variant}
        if not runs:
            continue
        name, r = max(runs.items(), key=lambda kv: kv[1]["dev_all"]["ndcg@10"])
        best_tt[variant] = r
        secs = sum(e["train_seconds"] for e in r["epochs"])
        add(
            label,
            {"all": r["dev_all"], "cold": r["dev_cold"]},
            secs,
            f"tau {r['tau']}, dim {r['dim']}, epoch {r['best_epoch']} (Kaggle GPU)",
        )

    n_cold = d["rows"]["popularity"]["cold"]["n_users"]
    out = [
        "# Recommender results (Office_Products, dev split)\n",
        f"Temporal leave-last-out per user: {info['users']:,} users, {info['items']:,} items, "
        f"{info['train_interactions']:,} train interactions. Metrics are **full ranking over all "
        f"{info['items']:,} items** (no sampled negatives), the user's train items excluded, "
        "ties counted against the model. Test split not used (Phase 7).\n",
        _table(
            [
                "Model",
                "HR@10",
                "NDCG@10",
                "Δ NDCG vs previous row",
                "Cold HR@10",
                "Cold NDCG@10",
                "Train s",
                "Setting",
            ],
            rows,
        ),
        "",
        f"Cold slice: the {n_cold:,} dev users whose target item has fewer than 5 train "
        "interactions (about 10% of users). Train seconds: popularity and ItemKNN on the local "
        "CPU, BPR on the local MPS GPU for its selected configuration, two-tower on a Kaggle GPU "
        "summed over all epochs run (before early stopping), so times are not comparable across "
        "hardware.\n",
    ]
    if len(best_tt) == 2:
        a, b = best_tt["id_only"], best_tt["id_text"]
        out += [
            "## Ablation: item tower ID-only vs ID + text embedding\n",
            _table(
                ["Item tower", "Best tau", "HR@10", "NDCG@10", "Cold HR@10", "Cold NDCG@10"],
                [
                    [
                        lbl,
                        str(r["tau"]),
                        f"{r['dev_all']['hr@10']:.4f}",
                        f"{r['dev_all']['ndcg@10']:.4f}",
                        f"{r['dev_cold']['hr@10']:.4f}",
                        f"{r['dev_cold']['ndcg@10']:.4f}",
                    ]
                    for lbl, r in (("ID-only", a), ("ID + text", b))
                ],
            ),
            "",
            f"Adding the text embedding changes overall NDCG@10 by "
            f"{b['dev_all']['ndcg@10'] - a['dev_all']['ndcg@10']:+.4f} and HR@10 by "
            f"{b['dev_all']['hr@10'] - a['dev_all']['hr@10']:+.4f}. On the cold slice both "
            "two-tower variants are near zero in absolute terms "
            f"(HR@10 {a['dev_cold']['hr@10']:.4f} and {b['dev_cold']['hr@10']:.4f}), far below "
            f"ItemKNN ({k['cold']['hr@10']:.4f}); the text signal helps overall ranking but does "
            "not rescue cold items in this setup. Likely cause (not verified): cold items keep "
            "an untrained random ID embedding that adds noise to their vector.\n",
        ]
    if tt:
        out += ["## Two-tower runs (dev, best epoch chosen by dev NDCG@10, patience 3)\n"]
        out.append(
            _table(
                ["Run", "Best epoch", "Epochs run", "HR@10", "NDCG@10"],
                [
                    [
                        n,
                        str(r["best_epoch"]),
                        str(len(r["epochs"])),
                        f"{r['dev_all']['hr@10']:.4f}",
                        f"{r['dev_all']['ndcg@10']:.4f}",
                    ]
                    for n, r in tt.items()
                ],
            )
        )
        out.append(
            "\nThe best temperature (0.1) is the largest tried, so the optimum may lie "
            "beyond the grid.\n"
        )
    out += [
        "## ItemKNN tuning (dev)\n",
        _table(
            ["k neighbours", "HR@10", "NDCG@10"],
            [
                [str(t["k"]), f"{t['all']['hr@10']:.4f}", f"{t['all']['ndcg@10']:.4f}"]
                for t in d["itemknn_tuning"]
            ],
        ),
        "\nThe best k is the largest tried (200), so the optimum may lie beyond the grid.\n",
    ]
    if "bpr_tuning" in d:
        out += [
            "## BPR tuning (20,000-user dev subset, NDCG@10)\n",
            _table(
                ["dim", "lr", "reg", "best epoch", "subset NDCG@10", "seconds"],
                [
                    [
                        str(t["dim"]),
                        str(t["lr"]),
                        str(t["reg"]),
                        str(t["best_epoch"]),
                        f"{t['subset_ndcg@10']:.4f}",
                        f"{t['train_seconds']:.0f}",
                    ]
                    for t in d["bpr_tuning"]
                ],
            ),
            "",
        ]
    (RESULTS / "recsys.md").write_text("\n".join(out))


if __name__ == "__main__":
    write_report()
