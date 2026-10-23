"""OpenAI-compatible chat clients (Groq, OpenRouter) and an ordered fallback chain.

Every failure (network, timeout, rate limit, bad HTTP status, malformed body) is raised as
`LLMError`, so callers can fall back to the regex-only parse.
"""

import json
import os
import time

import httpx

from cartrank import config


class LLMError(Exception):
    pass


class ChatClient:
    def __init__(
        self,
        name: str,
        url: str,
        model: str,
        key_env: str,
        extra: dict | None = None,
        timeout: float = config.LLM_TIMEOUT_S,
    ):
        self.name, self.url, self.model = name, url, model
        self.key_env, self.extra, self.timeout = key_env, extra or {}, timeout
        self.last_status: int | None = None

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        key = os.environ.get(self.key_env)
        if not key:
            raise LLMError(f"{self.key_env} is not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            **self.extra,
        }
        t0 = time.perf_counter()
        deadline = time.monotonic() + self.timeout  # total wall clock; httpx timeouts are per read
        try:
            with httpx.stream(
                "POST",
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {key}"},
            ) as r:
                self.last_status = r.status_code
                chunks = []
                for chunk in r.iter_bytes():
                    if time.monotonic() > deadline:
                        raise LLMError(f"{self.name}: deadline exceeded")
                    chunks.append(chunk)
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {type(e).__name__}") from e
        if time.monotonic() > deadline:
            raise LLMError(f"{self.name}: deadline exceeded")
        if r.status_code != 200:
            raise LLMError(f"{self.name}: HTTP {r.status_code}")
        try:
            body = json.loads(b"".join(chunks))
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.name}: malformed response") from e
        if not text:
            raise LLMError(f"{self.name}: empty content")
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "provider": self.name,
            "model": self.model,
            "api_seconds": time.perf_counter() - t0,
        }
        return text, usage


def groq(model: str = config.GROQ_MODEL, timeout: float = config.LLM_TIMEOUT_S) -> ChatClient:
    # gpt-oss are reasoning models: low effort keeps latency and tokens small for a parse task.
    return ChatClient(
        "groq",
        "https://api.groq.com/openai/v1/chat/completions",
        model,
        "GROQ_API_KEY",
        extra={"reasoning_effort": "low"},
        timeout=timeout,
    )


def openrouter(model: str, timeout: float = config.LLM_TIMEOUT_S) -> ChatClient:
    return ChatClient(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        model,
        "OPENROUTER_API_KEY",
        timeout=timeout,
    )


class FallbackClient:
    """Try clients in order; the first success wins. Raises LLMError if all fail."""

    def __init__(self, clients: list[ChatClient]):
        self.clients = clients
        self.errors: list[str] = []

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        self.errors = []
        for c in self.clients:
            try:
                text, usage = c.complete(system, user)
                usage["fallbacks_used"] = len(self.errors)
                return text, usage
            except LLMError as e:
                self.errors.append(str(e))
        raise LLMError("; ".join(self.errors))


def default_client(timeout: float = config.LLM_TIMEOUT_S, n_fallbacks: int | None = None):
    """Groq first, then OpenRouter free models. Returns None when no key is configured."""
    config.load_env()
    chain = []
    if os.environ.get("GROQ_API_KEY"):
        chain.append(groq(timeout=timeout))
    if os.environ.get("OPENROUTER_API_KEY"):
        models = config.OPENROUTER_FALLBACK_MODELS[:n_fallbacks]
        chain += [openrouter(m, timeout=timeout) for m in models]
    return FallbackClient(chain) if chain else None
