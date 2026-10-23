.PHONY: setup test lint fmt demo bundle quickstart

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: data profile

data:
	uv run python -m cartrank.data esci-splits
	uv run python -m cartrank.data reviews

profile:
	uv run python -m cartrank.data.profile

demo:
	uv run streamlit run app/streamlit_app.py

# Assemble artifacts/bundle (what would be uploaded to the Hugging Face Hub after owner approval)
bundle:
	uv run python -m cartrank.serve.artifacts

# Clean-machine path: install, lint, test; then the demo if the private artifacts are reachable.
quickstart: setup lint test
	@if [ -n "$$HF_TOKEN" ] || grep -q '^HF_TOKEN=' .env 2>/dev/null; then $(MAKE) demo; \
	else echo "Tests passed. To run the demo locally set HF_TOKEN (read access to the private artifacts repo) or use https://cartrank.streamlit.app"; fi
