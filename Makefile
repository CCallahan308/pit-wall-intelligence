.PHONY: help install ingest build test lint format dbt app api clean

help:
	@echo "Pit Wall Intelligence — make targets"
	@echo "  install  : install dependencies via uv"
	@echo "  ingest   : pull FastF1 data for the current season"
	@echo "  build    : run the full dbt pipeline"
	@echo "  test     : run pytest + dbt tests"
	@echo "  lint     : ruff + mypy"
	@echo "  format   : ruff format"
	@echo "  app      : launch Streamlit"
	@echo "  api      : launch FastAPI on :8000"
	@echo "  clean    : remove caches and build artifacts"

install:
	uv sync --extra dev

ingest:
	uv run python -m pitwall.ingest.fastf1_client --year 2024 --all

build:
	cd dbt && uv run dbt build

test:
	uv run pytest tests/ -v
	cd dbt && uv run dbt test

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

app:
	uv run streamlit run app/streamlit_app.py

api:
	uv run uvicorn api.main:app --reload --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf dbt/target dbt/logs dbt/dbt_packages
	find . -type d -name __pycache__ -exec rm -rf {} +
