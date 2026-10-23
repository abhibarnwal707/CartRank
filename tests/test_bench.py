from cartrank.serve.bench import run_benchmark
from cartrank.serve.sysinfo import info, rss_mb


class _R:
    def __init__(self, n):
        self.results = [0] * n
        self.timings_ms = {"parse": 1.0, "rerank": 10.0, "total": 11.0}


class _Pipe:
    def search(self, q):
        return _R(0 if q == "empty" else 3)


def test_run_benchmark_summarises():
    out = run_benchmark(_Pipe(), ["a", "b", "empty"])
    assert out["n_queries"] == 3 and out["empty_results"] == 1 and out["llm"] == "off"
    assert out["stages_ms_p50_p95"]["rerank"] == [10.0, 10.0]
    assert out["wall_ms"]["p50"] >= 0 and out["rss_mb"]["after"] > 0


def test_sysinfo():
    assert rss_mb() > 10 and {"platform", "python", "rss_mb"} <= set(info())
