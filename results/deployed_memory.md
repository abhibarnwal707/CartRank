# Deployed memory (Streamlit Community Cloud)

Resident memory of the live app's process, read from the app's own debug output after running the 50 demo
queries with the LLM off (2026-10-24). The cloud host is Linux x86_64 with Python 3.14.9.

| When | RSS MB |
|---|---|
| Before the 50 queries (models loaded) | 1,343 |
| After the 50 queries | 1,421 |

Budget 2,000 MB: within. Streamlit's documentation (read 2026-10-24, "as of February 2024") lists
2.7 GB as the free-tier memory ceiling. Latency is intentionally not reported for the free-tier
deployment. Source: `results/deployed_memory.json`.
