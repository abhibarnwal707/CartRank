"""Build the real Pipeline from artifacts. Used by the Streamlit app and the latency script."""

import os
import time

from cartrank.query.llm_clients import default_client
from cartrank.recsys.serving import TwoTowerServing
from cartrank.search.catalog_index import CatalogSearch
from cartrank.serve.artifacts import resolve
from cartrank.serve.pipeline import Pipeline

APP_LLM_TIMEOUT_S = 3.5  # hard per-call deadline, so a throttled free tier cannot hang the UI
APP_LLM_FALLBACKS = 1


def load_pipeline(device: str | None = "cpu") -> Pipeline:
    t0 = time.perf_counter()
    p = resolve()
    t_resolve = time.perf_counter() - t0  # includes the Hub download on a cold machine
    t1 = time.perf_counter()
    cs = CatalogSearch(
        "ft",
        str(p.encoder),
        p.hybrid_k,
        p.hybrid_depth,
        catalog_path=p.catalog,
        vecs_path=p.vecs,
        device=device,
    )
    tt = TwoTowerServing.load(p.serving)
    # CARTRANK_NO_LLM=1 forces regex-only mode (tests, and a way to demo without API keys)
    client = None
    if not os.environ.get("CARTRANK_NO_LLM"):
        client = default_client(timeout=APP_LLM_TIMEOUT_S, n_fallbacks=APP_LLM_FALLBACKS)
    pipe = Pipeline(cs, tt, client)
    pipe.startup = {
        "artifacts_s": round(t_resolve, 1),
        "load_s": round(time.perf_counter() - t1, 1),
    }
    return pipe
