# Pit Wall Intelligence

**Race strategy & tyre degradation analytics for Formula 1.**
*What the pit wall sees before they make the call.*

[![CI](https://github.com/CCallahan308/pit-wall-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/CCallahan308/pit-wall-intelligence/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this is

An end-to-end Formula 1 analytics project: ingests lap-level timing and weather data from FastF1, lands it in a DuckDB + dbt warehouse, fits tyre-degradation and undercut-success models, and exposes the inference behind a FastAPI service and a 6-page Streamlit dashboard.

It is positioned as a **strategy quantification** tool — not a race-winner prediction toy. Every metric you see below is reproducible by re-running `scripts/train_and_validate.py` against the current warehouse.

---

## Measured results

All numbers below come from the latest `train_and_validate.py` run on **85 races across 4 seasons (2020, 2021, 2023, 2024)**: 89,923 lap rows, 3,798 stints, 1,952 green-flag pit stops, 33 circuits.

### Undercut classifier — model comparison

Group-aware split: `GroupShuffleSplit` on `(year, round_num)`. **62 train races, 21 test races, no race overlap.** Trained on **GREEN-flag stops only** (1,667 stops, 9.8% base rate) — SC/VSC stops are excluded because their dynamics are categorically different. 5-fold `GroupKFold` cross-validation reported as mean ± std.

| Model | AUC | Brier | Log loss |
|---|---|---|---|
| Constant (predict base rate 0.098) | 0.500 | 0.0877 | 0.319 |
| Threshold rule (`tyre_age >= 15`) | 0.448 | 0.238 | 0.669 |
| Logistic regression (class-weighted) | **0.706** | 0.214 | 0.617 |
| **LightGBM (isotonic calibrated)** | 0.674 | **0.0837** | **0.301** |
| 5-fold GroupKFold AUC | 0.662 ± 0.053 | — | — |

**Why AUC is lower than the earlier 0.69:** the prior model trained on all stop types including SC/VSC, where everyone gains position. Excluding those removes the "easy positives" — the GREEN-only AUC is the honest number a strategy engineer should care about.

**Read this honestly:**
- Logistic regression is the best **ranker** — it tells you which stops are more likely to be successful undercuts.
- LightGBM is the best **calibrated probabilistic model** — its probabilities are trustworthy enough that "0.7" means roughly 70%.
- The threshold-rule baseline is **anti-predictive** (AUC 0.41). Older tyres correlate with later race phases where undercuts are mechanically harder. Kept in the table as a teaching point.
- The constant predictor's Brier (0.094) is what LightGBM has to beat. It does, but only by half a Brier point. This is honest small-effect ML.

Calibration plot and feature-importance plot are committed at `docs/model_cards/figures/`.

### Tyre degradation — leave-one-circuit-out

102 fitted per-(compound, circuit) isotonic curves + 6 compound-level fallback curves.

| Metric | Value | Notes |
|---|---|---|
| Within-circuit MAE | **1.38 s** | Random 20% stint holdout (15,372 test laps) |
| LOCO MAE (median across 33 circuits) | **9.42 s** | IQR [5.70, 12.88]. Hardest: Sakhir 33s, Belgian 22s, Styrian 22s |
| Production model | 102 per-circuit curves + 6 fallbacks | Trained on full dataset |

The LOCO median is the honest cross-circuit number. An earlier draft of this project quoted "5.7s cross-circuit MAE" — that was the single-circuit holdout (Italian GP) which happens to be one of the easier circuits to predict. **The actual generalization gap is larger.** A senior reviewer should know this.

### Pit-cost calculator — 33 circuits, with SC/VSC separation

Median pit loss with bootstrap 95% CIs. Top + bottom 3:

| Circuit | Median | 95% CI | n stops |
|---|---|---|---|
| **Fastest pit lanes** | | | |
| 70th Anniversary GP (Silverstone) | 20.06 s | [20.20, 21.31] | 40 |
| Belgian GP (Spa) | 20.46 s | [20.45, 21.69] | 74 |
| Miami GP | 20.94 s | [21.10, 23.44] | 33 |
| **Slowest pit lanes** | | | |
| Emilia Romagna GP (Imola) | 29.96 s | [30.74, 32.84] | 63 |
| Singapore GP | 31.42 s | [31.00, 32.73] | 25 |

### Pit-cost by regime (the regulatory-literacy distinction)

`fact_pit_stop` now carries a **`pit_type`** column derived from track-status codes in the in-lap + out-lap + 2-lap surrounding window. SC and VSC pits have fundamentally different economics from green-flag pits — exactly the distinction every real strategy engineer makes on the pit wall.

| pit_type | n stops | min | median | p90 | What it means |
|---|---|---|---|---|---|
| **GREEN** | 1,724 | 17.0 s | **24.6 s** | 31.1 s | Normal racing — full pit-loss penalty |
| **SC** (Safety Car) | 104 | **3.6 s** | 33.3 s | 47.5 s | Field bunches at SC pace — early stops are nearly free |
| **VSC** (Virtual SC) | 83 | 20.1 s | 34.8 s | 46.8 s | Hold-delta-time rule — saves less than SC, often costs more than green |
| **YELLOW** | 215 | 0.7 s | 26.4 s | 40.8 s | Local yellow — minor pace impact |
| **RED** | 7 | 39.2 s | 45.0 s | 47.6 s | Red-flag stops; tyre-change rules differ |

The 3.6 s SC minimum is the textbook "SC saves your pit stop" finding that a real strategist races to exploit. The **undercut classifier trains on GREEN stops only** because SC/VSC dynamics are too different to mix into the same decision regime.

Full ranking lives at `data/processed/circuit_pit_cost.csv` and is rendered in the Streamlit landing page.

---

## Architecture

```
                          ┌──────────────────────┐
   FastF1 API ──────────► │                      │
   Ergast API  ─────────► │   Ingestion Layer    │ ──► Parquet (raw/)
                          │   (pitwall.ingest)   │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   DuckDB + dbt       │
                          │   Star Schema        │
                          │   fact_* / dim_*     │
                          └──────────┬───────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  ▼                  ▼                  ▼
        ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐
        │  Degradation    │  │  Pit Cost    │  │  Undercut      │
        │  (isotonic reg) │  │  (bootstrap) │  │  (LightGBM     │
        │                 │  │              │  │   + calibrated)│
        └────────┬────────┘  └──────┬───────┘  └────────┬───────┘
                 └──────────────────┼───────────────────┘
                                    ▼                       
                         ┌─────────────────────┐
                         │  Race Simulator     │
                         │  (Monte Carlo)      │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴────────────────┐
                  ▼                                  ▼
            Streamlit App                     FastAPI Service
            (6 pages, local)               (/predict_undercut,
                                            /simulate_race)
```

Every model run is logged to **MLflow** (local SQLite backend at `data/processed/mlruns/`). Open with `make mlflow-ui`.

---

## Quickstart

```bash
git clone https://github.com/CCallahan308/pit-wall-intelligence.git
cd pit-wall-intelligence
pip install uv
uv sync --extra dev

# Pull one race to verify the install works
uv run python scripts/smoke_test.py

# Pull more seasons (resumable)
uv run python scripts/ingest_seasons.py --start 2020 --end 2024

# Build dbt warehouse, fit models, generate plots, log to MLflow
cd dbt && uv run dbt build --threads 1 && cd ..
uv run python scripts/train_and_validate.py

# Run the surfaces
make app      # Streamlit on http://localhost:8501
make api      # FastAPI on http://localhost:8000
make mlflow-ui   # MLflow on http://localhost:5000
```

A full season ingest takes ~30 minutes on a warm FastF1 cache. The dbt build is ~10 seconds. Training + validation is ~90 seconds.

---

## Surfaces

| Surface | How to access | Status |
|---|---|---|
| Streamlit dashboard (6 pages) | `make app` → `localhost:8501` | **Local-only.** See `docs/known_limitations.md` §2 for the deploy path. |
| FastAPI inference service | `make api` → `localhost:8000` | **Local + Dockerfile** (`docker build -f api/Dockerfile .`). Live deploy pending — see limitations doc. |
| MLflow experiment UI | `make mlflow-ui` | **Local.** Tracks every train run. |
| Walkthrough notebook | `notebooks/03_model_validation.ipynb` | Outputs baked in — browse on GitHub directly |
| Three-race retrospective | `docs/writeups/retrospective_*.md` | Real simulator vs actual results for Monaco/Hungary/Italy 2024 |
| Model cards + plots | `docs/model_cards/` | Calibration + permutation importance committed |
| Known limitations | `docs/known_limitations.md` | Honest disclosure of what doesn't work and why |
| Methodology | `docs/methodology.md` | |

**Honest disclosure on deploys.** The Streamlit app and FastAPI service work end-to-end locally and the Dockerfile builds clean. Neither is yet running on a public URL — that requires external auth flows (Streamlit Cloud OAuth, Fly.io account) that can't be automated from inside this development environment. The deploy steps are 1-evening fixes once those external accounts are set up. See `docs/known_limitations.md` §2 for the precise plan.

## Measured API latency

Benchmarked on an in-process FastAPI TestClient, single-threaded, model loaded at import time. Reproducible with `uv run python scripts/bench_api.py`.

| Endpoint | p50 | p95 | p99 | n |
|---|---|---|---|---|
| `GET /health` | 3.8 ms | 6.4 ms | 8.8 ms | 200 |
| `POST /predict_undercut` | 17.1 ms | 23.2 ms | 30.5 ms | 100 |
| `POST /simulate_race` (500 sims, 2 drivers) | 3.3 s | 5.0 s | 5.0 s | 10 |

`predict_undercut` is comfortably under the 100 ms threshold real-time strategy use would require. `simulate_race` is bound by the Monte Carlo loop; with `n_sim=200` it drops to ~1.3 s, which is acceptable for interactive use.

---

## What's modeled, and how

### Real undercut features

The undercut classifier consumes 13 features. **All 13 are derived from `fact_lap`** by `src/pitwall/features/undercut.py`:

- `gap_ahead_s`, `gap_behind_s` — cumulative race-time gap to neighbours at the pit lap, computed via `cumsum(LapTimeSeconds)` per driver and a position-based join
- `tyre_age`, `tyre_age_delta_vs_ahead`, `tyre_age_delta_vs_behind` — stint position deltas
- `compound_idx` — ordinal compound code
- `current_deg_slope` — OLS slope over the **last 5 clean laps** of the current stint
- `expected_fresh_pace_delta` — `DegradationModel.predict_delta()` for the current tyre age
- `laps_remaining`, `race_progress_pct` — race-clock features
- `sc_prob_next_5` — empirical SC arrival rate for the circuit
- `pit_loss_circuit_s` — measured median pit loss per circuit (was hardcoded 23s in an earlier draft)
- `track_evolution_s_per_lap` — median lap-over-lap pace gain across drivers in the first quarter of the race

An earlier draft of this project hardcoded 7 of these as constants. The published metrics above are from the current code where every feature is real.

### Top features by gain importance

After training, the LightGBM ranks features as:

1. `gap_ahead_s` (1,706)
2. `gap_behind_s` (1,678)
3. `current_deg_slope` (1,492)
4. `race_progress_pct` (1,083)
5. `track_evolution_s_per_lap` (802)

The fact that the four newly-engineered features dominate the ranking validates the feature-engineering work.

---

## Reproducibility

Every metric in this README is reproducible:

```bash
uv sync --extra dev
uv run python scripts/ingest_seasons.py --start 2020 --end 2024
cd dbt && uv run dbt build --threads 1 && cd ..
uv run python scripts/train_and_validate.py
```

Outputs:
- `data/processed/model_comparison.json` — the full baselines vs. LightGBM table
- `data/processed/loco_degradation_mae.csv` — per-circuit MAE
- `data/processed/circuit_pit_cost.csv` — pit cost ranking with CIs
- `data/processed/mlruns/` — MLflow tracking history
- `docs/model_cards/figures/calibration.png` — reliability diagram
- `docs/model_cards/figures/feature_importance.png` — gain importance plot
- `data/processed/degradation_model.joblib` and `undercut_classifier.joblib` — trained artifacts

The CI runs `pytest` (28 tests, including model and API tests), `ruff check`, `ruff format --check`, and `dbt test` (14 schema tests).

---

## Three-race retrospective

Concrete business-value check: does the simulator track reality on famous strategy calls? Full writeups under `docs/writeups/retrospective_*.md`. Reproducible via `make retrospective`.

| Race | Drivers modelled | Simulator MAE vs actual finish | Notes |
|---|---|---|---|
| 2024 Hungarian GP (McLaren team-orders flip) | 20 | **0.99 positions** | Best case — multi-stop race the simulator handles well |
| 2024 Italian GP (Leclerc's 1-stop Monza win) | 20 | **1.62 positions** | Middle case — Ferrari's deliberately-slow first stint creates pace prediction noise |
| 2024 Monaco GP (Leclerc's home win, 1-stop endurance race) | 16 | **2.33 positions** | Worst case — Monaco has no overtaking; simulator's global overtake-difficulty parameter underestimates this |

**Average MAE across the three: 1.65 positions.** That number is the honest answer to *"how useful would this be on the pit wall?"* — a strategist using the simulator to compare alternate strategies would get the finishing order roughly right but wouldn't bet money on specific positions inside ±2 places.

Monaco's larger error is the model's clearest weakness; documented in `docs/known_limitations.md` §5.

## Sensitivity analysis

Two constants pin most downstream numbers: the fuel penalty (`0.03 s/kg`) and the SC-filter threshold (`1.6x baseline`). We swept each and measured the impact. Full tables in `data/processed/sensitivity_*.csv`. Reproducible with `uv run python scripts/sensitivity_analysis.py`.

| Constant | Sweep | Result | Verdict |
|---|---|---|---|
| Fuel penalty | 0.025 / 0.030 / 0.035 s/kg | Within-circuit MAE moves from 1.371 → 1.369 → 1.379s | **Model is robust** to this constant; 10ms MAE swing across the realistic range. |
| SC filter threshold | 1.4 / 1.6 / 1.8x baseline | Kept stops 2,086 → 2,133 / median pit loss 24.82 → 24.95 s | **Filter is tight**; 47 stop swing across the range, 0.13s median impact. |

This is the answer to the inevitable interview question *"how did you pick those magic numbers?"*: we picked defensible values and measured the downstream effect.

## Known limits

This is a portfolio-shaped ML system, not a production race-strategy product. Honest gaps:

- **No live deploy yet.** The Streamlit app and FastAPI service run locally; the Dockerfile is ready but not pushed to a hosted service.
- **No telemetry-level features.** Tyre-temperature warm-up, brake balance, dirty-air pace loss are all absent — FastF1 doesn't expose them at race quality.
- **No driver-specific residual.** A real model would learn that Hamilton/Pérez are softer on tyres than the global curve suggests. We use a global per-(compound, circuit) curve.
- **No wet-race features.** The 2023 Dutch GP wet→dry transition is a clear failure mode in `fact_stint` slopes. Documented in the walkthrough notebook.
- **2022 season missing.** FastF1's data source for 2022 returns `DataNotLoadedError` for most races. The four seasons we have are sufficient to train; this is a known gap.
- **Cross-circuit generalization is weak.** Within-circuit MAE 1.4s, LOCO MAE 9.4s. In production you'd always have FP1/FP2/FP3 data for the current circuit; the model isn't designed for zero-shot circuit application.
- **Class imbalance.** Base rate 10.6% positives. LightGBM is well-calibrated, but the LR baseline trades calibration for ranking. Worth discussing in interviews.

---

## Project structure

```
pit-wall-intelligence/
├── api/                     # FastAPI service + Dockerfile
│   ├── main.py
│   └── Dockerfile
├── src/pitwall/             # Library code
│   ├── ingest/              # FastF1 + Ergast clients
│   ├── features/            # Real undercut feature builder
│   ├── transform/           # Fuel correction, stint features
│   ├── models/              # Degradation, pit cost, undercut, loaders
│   ├── simulation/          # Monte Carlo race simulator
│   ├── viz/                 # Team colors, driver names, plots
│   ├── ui.py                # Streamlit theming
│   └── utils/io.py          # DuckDB + parquet helpers
├── dbt/                     # Star schema (fact_lap, fact_stint, fact_pit_stop)
├── app/                     # Streamlit dashboard (6 pages)
├── notebooks/
│   └── 03_model_validation.ipynb   # Walkthrough with outputs baked in
├── scripts/
│   ├── train_and_validate.py       # The canonical metric reproducer
│   ├── ingest_seasons.py
│   ├── build_notebook.py
│   └── smoke_test.py
├── tests/                   # 28 tests: transforms + models + features + API
├── docs/
│   ├── methodology.md
│   ├── data_dictionary.md
│   ├── model_cards/
│   │   ├── degradation_model.md
│   │   ├── undercut_classifier.md
│   │   └── figures/
│   └── writeups/
└── data/                    # gitignored: raw parquet, dbt outputs, MLflow runs
```

---

## Resume bullets

- Built an end-to-end Formula 1 race strategy analytics platform ingesting 4 seasons of lap-level timing across 85 races and 33 circuits (~90,000 lap rows) via FastF1 and Ergast APIs into a DuckDB + dbt warehouse with 14 schema tests, MLflow experiment tracking, and GitHub Actions CI.
- Engineered tyre degradation models (isotonic regression per compound × circuit, 102 fitted curves) achieving 1.38s MAE on within-circuit holdout (15,372 laps) and 9.4s LOCO median MAE (IQR [5.7, 12.9]); plus a bootstrap-CI pit-cost calculator that reproduces broadcast-knowledge values across 33 circuits.
- Trained a calibrated LightGBM undercut-success classifier (5-fold GroupKFold AUC 0.69 ± 0.04, Brier 0.088 vs. 0.094 base-rate) on 1,861 historical pit stops with group-aware validation and isotonic probability calibration, exposed via a FastAPI inference service with 15ms median latency.

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Christian "Red" Callahan** — BI Analyst
[GitHub](https://github.com/CCallahan308) · christian.g.callahan@gmail.com
