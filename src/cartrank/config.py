"""Local configuration: .env loading and the pinned LLM models (Phase 5)."""

import os

from cartrank.data.paths import ROOT

# Pinned in Phase 5 after listing each provider's models and test-calling candidates on
# 2026-10-24. Free OpenRouter models are flaky (upstream 429/404), so this is an ordered chain.
# Tried and dropped that day: nemotron-3-super-120b (404 upstream), liquid/lfm-2.5-2.6b (~59 s).
GROQ_MODEL = "openai/gpt-oss-20b"
OPENROUTER_FALLBACK_MODELS = (
    "recursive/flint-v1.0:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
)
LLM_TIMEOUT_S = 15.0


def load_env() -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing variables win)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
