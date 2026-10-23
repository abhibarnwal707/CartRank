"""Locate the runtime artifacts: local repo layout, a flat bundle directory, or the HF Hub.

Bundle layout (what `make bundle` writes and what gets uploaded, after owner approval):
  catalog.parquet  catalog_vecs.npy  encoder/  serving/{two_tower_serving.npz,item_ids.txt}
  config.json      {"hybrid_k": ..., "hybrid_depth": ...}
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from cartrank.data.catalog import SLIM_PATH
from cartrank.data.paths import ROOT

BUNDLE = ROOT / "artifacts" / "bundle"
HF_ARTIFACT_REPO: str | None = None  # set once the owner approves creating the Hub repo
ENCODER_LOCAL = ROOT / "artifacts" / "models" / "bge-small-esci-ft"
VECS_LOCAL = ROOT / "artifacts" / "catalog" / "ft_vecs.npy"
SERVING_LOCAL = ROOT / "artifacts" / "serving"


@dataclass
class Paths:
    catalog: Path
    vecs: Path
    encoder: Path
    serving: Path
    hybrid_k: int
    hybrid_depth: int


def bundle_paths(root: Path) -> Paths:
    cfg = json.loads((root / "config.json").read_text())
    return Paths(
        root / "catalog.parquet",
        root / "catalog_vecs.npy",
        root / "encoder",
        root / "serving",
        cfg["hybrid_k"],
        cfg["hybrid_depth"],
    )


def local_paths() -> Paths:
    from cartrank.search.catalog_index import hybrid_setting

    k, depth = hybrid_setting("ft")
    return Paths(SLIM_PATH, VECS_LOCAL, ENCODER_LOCAL, SERVING_LOCAL, k, depth)


def resolve() -> Paths:
    """CARTRANK_ARTIFACT_DIR > local repo artifacts > Hugging Face Hub."""
    env = os.environ.get("CARTRANK_ARTIFACT_DIR")
    if env:
        return bundle_paths(Path(env))
    if BUNDLE.exists() and (BUNDLE / "config.json").exists():
        return bundle_paths(BUNDLE)
    if SLIM_PATH.exists() and ENCODER_LOCAL.exists():
        return local_paths()
    if HF_ARTIFACT_REPO:
        from huggingface_hub import snapshot_download

        return bundle_paths(Path(snapshot_download(HF_ARTIFACT_REPO, repo_type="model")))
    raise FileNotFoundError("no artifacts found: build them or set CARTRANK_ARTIFACT_DIR")


def build_bundle(out: Path = BUNDLE) -> None:
    p = local_paths()
    if out.exists():
        shutil.rmtree(out)
    (out / "serving").mkdir(parents=True)
    shutil.copy(p.catalog, out / "catalog.parquet")
    shutil.copy(p.vecs, out / "catalog_vecs.npy")
    shutil.copytree(p.encoder, out / "encoder")
    for f in ("two_tower_serving.npz", "item_ids.txt"):
        shutil.copy(p.serving / f, out / "serving" / f)
    (out / "config.json").write_text(
        json.dumps({"hybrid_k": p.hybrid_k, "hybrid_depth": p.hybrid_depth})
    )
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"bundle at {out}: {total:.0f} MB")


if __name__ == "__main__":
    build_bundle()
