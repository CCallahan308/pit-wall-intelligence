# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management and a `Makefile` as the canonical task runner.

```bash
# Install (creates .venv and resolves uv.lock)
uv sync --extra dev

# Ingest data — Round-level smoke test, then full season
uv run python -m pitwall.ingest.fastf1_client --year 2024 --round 8
uv run python -m pitwall.ingest.fastf1_client --year 2024 --all   # or: make ingest

# Build dbt warehouse (DuckDB target at data/processed/pitwall.duckdb)
cd dbt && uv run dbt build && cd ..                              # or: make build

# Tests — pytest then dbt tests (both must pass for CI green)
uv run pytest tests/ -v                                          # python tests
uv run pytest tests/test_pit_cost.py::test_name -v               # single test
cd dbt && uv run dbt test && cd ..                               # 14 schema tests
make test                                                        # runs both

# Lint / format — CI fails on either ruff check or ruff format --check
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/                           # CI uses --check
uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/   # make format
uv run mypy src/                                                 # informational, not blocking

# Run surfaces
uv run streamlit run app/streamlit_app.py                        # make app
uv run uvicorn api.main:app --reload --port 8000                 # make api (stretch)

# End-to-end pipeline validation against ingested data — produces the README metrics
uv run python scripts/train_and_validate.py
```

## Architecture

The pipeline is a one-way flow: **FastF1/Ergast → Parquet → DuckDB (via dbt) → ML models → Streamlit/FastAPI**. Understanding the layers and how they share data is the key to being productive here.

### 1. Ingestion → partitioned Parquet

`src/pitwall/ingest/fastf1_client.py` pulls each session via FastF1 and writes laps to `data/raw/laps/year={Y}/round={R}/session={S}.parquet` (hive-partitioned). The FastF1 cache lives in `cache/` — re-runs are cheap, cold runs take ~45 min for a full season. Weather and results sit beside laps under `data/raw/`.

### 2. dbt + DuckDB warehouse

`dbt/profiles.yml` points at `../data/processed/pitwall.duckdb`. The `stg_laps` view reads the partitioned parquet directly via `read_parquet(..., hive_partitioning=true)` — there is no separate "load into DB" step. Models flow `staging (view) → intermediate (view) → marts (table)`:

- `fact_lap` — one row per driver-lap, fuel-corrected pace, cleanliness flags
- `fact_stint` — one row per (race, driver, stint) with degradation slope
- `fact_pit_stop` — one row per stop with `pit_loss_s` vs. clean-air baseline

Schema tests in `dbt/models/marts/schema.yml` enforce realistic value ranges (e.g. `lap_time_s` between 50–300s, `pit_loss_s` between 3–50s). **Schema drift fails CI.**

Important: FastF1 timing columns arrive as nanosecond `BIGINT` (pandas `timedelta64[ns]` round-trip through parquet). `stg_laps` divides by `1e9` to get seconds — preserve this convention in any new staging models.

### 3. ML models — load from DuckDB, not Parquet

All four models in `src/pitwall/models/` and `src/pitwall/simulation/` consume the `fact_*` tables from DuckDB. The flow is:

- **Fuel correction** (`transform/fuel_correction.py`) — subtracts `~0.03s × kg_remaining` before any degradation analysis. Constants live in `src/pitwall/config.py` (`FUEL_BURN_KG_PER_LAP`, `FUEL_TIME_PENALTY_S_PER_KG`, `STARTING_FUEL_KG`). If you change these you change every downstream metric.
- **Degradation curve** (`models/degradation_curve.py`) — isotonic regression per `(compound, circuit)`. Falls back to a compound-only curve on cross-circuit residuals when a `(compound, circuit)` cell is under-sampled. Has two prediction modes: `predict_pace` (absolute) and `predict_delta` (vs. fresh tyre).
- **Pit cost** (`models/pit_cost.py`) — median `(in-lap + out-lap)` delta vs. the driver's clean-air baseline (gap-ahead > `CLEAN_AIR_GAP_S = 2.0s`), per circuit.
- **Undercut classifier** (`models/undercut_classifier.py`) — calibrated LightGBM, isotonic-calibrated probabilities. Feature list is `FEATURE_COLS` — keep in sync if you add features.
- **Race simulator** (`simulation/race_simulator.py`) — Monte Carlo, takes `RaceConfig` + `DriverPlan` list, branches on Safety Car (Poisson) and pit-loss variance.

`scripts/train_and_validate.py` is the canonical "does the whole pipeline still work" check — it loads from DuckDB, trains all models, and prints the metrics in the README. Run it after any modeling change.

### 4. Surfaces

`app/streamlit_app.py` is the entry; pages under `app/pages/` auto-register via Streamlit's multipage convention (numeric prefixes drive sort order). `api/` is a stretch FastAPI service — currently scaffolding only.

## Conventions worth knowing

- **uv-only** — `pip install` directly into the venv breaks `uv.lock` reproducibility. Always `uv add <pkg>` or edit `pyproject.toml` then `uv sync`.
- **Ruff config has ML-aware ignores** in `pyproject.toml`: `N803`/`N806` are off so `X`, `X_train`, `X_test` are legal. `RUF002`/`RUF003` are off so en-dashes and `×` in docstrings/comments don't trigger.
- **Line length 100**, not 88. Targeted at `py311`.
- **Modeling PR checklist** (from CONTRIBUTING.md): updating the degradation or undercut models requires updating the matching model card in `docs/model_cards/`, re-running `notebooks/03_model_validation.ipynb`, and noting the metric delta in the PR description.
- `data/`, `cache/`, and `dbt/target/` are gitignored — never check in built artifacts or the DuckDB file.
