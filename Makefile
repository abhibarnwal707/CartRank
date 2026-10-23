.PHONY: setup test lint fmt demo bundle

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
