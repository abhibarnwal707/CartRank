"""Dense retrieval: sentence-transformers encoder + FAISS flat inner-product index."""

import sys
import time
from pathlib import Path

import faiss
import numpy as np

from cartrank.data.paths import ROOT
from cartrank.search.corpus import DEFAULT_VARIANT, load_corpus

BASE_MODEL = "BAAI/bge-small-en-v1.5"
# Query instruction from the bge-small-en-v1.5 model card.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 128
ARTIFACTS = ROOT / "artifacts" / "dense"


def vectors_path(tag: str, variant: str) -> Path:
    return ARTIFACTS / f"{tag}_{variant}.npy"


def load_encoder(model: str = BASE_MODEL):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = SentenceTransformer(model, device=device)
    enc.max_seq_length = MAX_SEQ_LENGTH
    return enc


def encode_queries(enc, queries: list[str], prefix: str = BGE_QUERY_PREFIX, batch_size: int = 256):
    return enc.encode(
        [prefix + q for q in queries],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_corpus(tag: str, model: str, variant: str, ids: list[str] | None = None) -> None:
    corpus = load_corpus()
    if ids is not None:
        corpus = corpus[corpus.product_id.isin(set(ids))]
    enc = load_encoder(model)
    texts = corpus[f"text_{variant}"].tolist()
    t0 = time.time()
    vecs = enc.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path(tag, variant), vecs)
    print(f"embedded {len(texts)} texts in {time.time() - t0:.0f}s -> {vecs.shape}")


def flat_ip_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


if __name__ == "__main__":
    # python -m cartrank.search.dense <tag> <model> [variant]
    embed_corpus(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_VARIANT)
