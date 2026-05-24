# Pit Wall Intelligence

**Race strategy & tyre degradation analytics for Formula 1.**
*What the pit wall sees before they make the call.*

[![CI](https://github.com/CCallahan308/pit-wall-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/CCallahan308/pit-wall-intelligence/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](#)

---

## What this is

Pit Wall Intelligence is an end-to-end Formula 1 analytics platform that reconstructs race strategy decisions and quantifies their cost in tenths of a second. It models tyre degradation per compound and circuit, computes pit stop loss curves, identifies undercut/overcut windows, and simulates alternate strategies — the same questions a real race strategist asks on the pit wall.

Unlike most public F1 projects, this is **not a race-winner prediction toy**. It is a *strategy quantification* tool, modelled on how Mercedes, McLaren, Ferrari, and Pirelli actually use lap-level data.

## The hero question this answers

> *"At lap 22, Pérez is 1.8s behind Hamilton on tyres that are 8 laps older. If we pit now, do we gain or lose track position by lap 30?"*

This repo gives you a calibrated, explainable answer in under 200ms.

---

## Highlights

Numbers below are **measured on 85 races across 4 seasons** (2020, 2021, 2023, 2024 — 89,923 lap rows, 3,962 stints, 2,029 green-flag pit stops across 33 circuits). All metrics are reproducible via `scripts/train_and_validate.py`.

- **DuckDB + dbt** warehouse with a proper star schema (`fact_lap`, `fact_stint`, `fact_pit_stop`), all 14 schema tests passing
- **Tyre degradation curves** via isotonic regression per compound × circuit (102 fitted curves):
  - **1.38 s MAE** on within-circuit holdout (15,372 test laps)
  - **5.70 s MAE** on cross-circuit holdout (Italian GP held out entirely) — the model needs *some* practice data from a circuit to calibrate its baseline, which teams always have via FP1/2/3
- **Pit-cost calculator** with bootstrap 95% CIs across 33 circuits — Singapore correctly identified as the slowest pit lane (31.4s), Spa the fastest (20.5s), Bahrain 25.9s, Monza 26.6s, Monaco 23.0s — all aligned with broadcast-knowledge values
- **Calibrated LightGBM undercut classifier**: AUC **0.701**, Brier **0.093** on a 466-stop test split (1,861 total examples, 10.6% positive base rate)
- **Monte Carlo race simulator** with per-lap pace noise, pit-loss variance, and Poisson Safety Car arrivals — produces realistic finishing-position distributions
- **Streamlit dashboard** with 6 pages (landing, stint analysis, pit window finder, driver comparison, strategy simulator, season overview, race-by-race) — all backed by the trained models
- Automated **GitHub Actions CI** (pytest 8/8 passing, dbt test 14/14 passing, ruff)

---

## Live demo

| Surface | Link |
|---|---|
| Streamlit app | _local-only for now (`make app`) — deploy to Community Cloud once full-season data is loaded_ |
| Strategy API | _stretch goal_ |
| LinkedIn writeup | [docs/writeups/LINKEDIN_POST.md](docs/writeups/LINKEDIN_POST.md) — held until full-season run |
| Methodology | [docs/methodology.md](docs/methodology.md) |
| Pipeline validation script | [scripts/train_and_validate.py](scripts/train_and_validate.py) — produces the numbers above |

---

## Architecture

```
                          ┌──────────────────────┐
   FastF1 API ──────────► │                      │
   Ergast API  ─────────► │   Ingestion Layer    │ ──► Parquet (raw/)
   Open-Meteo  ─────────► │   (pitwall.ingest)   │
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
        │  Model          │  │  Calculator  │  │  Classifier    │
        │  (isotonic reg) │  │              │  │  (LightGBM)    │
        └────────┬────────┘  └──────┬───────┘  └────────┬───────┘
                 └──────────────────┼───────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Race Simulator     │
                         │  (Monte Carlo)      │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼────────────────┐
                  ▼                 ▼                ▼
            Streamlit App     FastAPI Service    Power BI
```

---

## Quickstart

```bash
# 1. clone
git clone https://github.com/CCallahan308/pit-wall-intelligence.git
cd pit-wall-intelligence

# 2. install (uses uv — fast, deterministic)
pip install uv
uv sync

# 3. pull a single race for a smoke test
uv run python -m pitwall.ingest.fastf1_client --year 2024 --round 8

# 4. build the dbt models
cd dbt && uv run dbt build && cd ..

# 5. launch the dashboard
uv run streamlit run app/streamlit_app.py
```

Full ingestion (5 seasons) takes ~45 minutes on a warm FastF1 cache.

---

## What's modeled, and how

### 1. Fuel correction
Lap times are normalised by subtracting an estimated fuel-load effect (`~0.03s per lap per kg`). Without this, "tyre degradation" is contaminated by fuel burn-off. See [`src/pitwall/transform/fuel_correction.py`](src/pitwall/transform/fuel_correction.py).

### 2. Tyre degradation
Isotonic regression of fuel-corrected lap time vs. tyre age, fit per `(compound, circuit, stint_position)`. Enforces monotonic degradation, robust to outliers (traffic, mistakes, dirty laps). Bootstrap CIs around each curve.

### 3. Pit stop cost
Median (in-lap + out-lap) delta vs. driver's clean-air green-lap baseline, computed per circuit. This is the single number every strategist obsesses over — typically **18–25s** depending on pit lane length and speed limit.

### 4. Undercut classifier
LightGBM binary classifier: *"given current race state, does pitting NOW gain net positions by lap N+5?"* Features include gap-ahead, gap-behind, tyre age delta, compound delta, current degradation slope, track evolution, and recent SC probability. Isotonic-calibrated probabilities. AUC 0.82, Brier 0.16 on 2024 holdout.

### 5. Race simulator
Monte Carlo: 10k simulations per scenario, branching on weather windows (Open-Meteo), Safety Car arrival (Poisson per circuit), and pit window timing. Outputs distribution of finishing positions per strategy choice.

---

## Project structure

```
pit-wall-intelligence/
├── src/pitwall/        # Library code
│   ├── ingest/         # FastF1, Ergast, weather clients
│   ├── transform/      # Fuel correction, stint features, degradation
│   ├── models/         # Degradation curve, pit cost, undercut classifier
│   ├── simulation/     # Monte Carlo race simulator
│   ├── viz/            # Team colors, hero charts
│   └── utils/          # Cache management
├── dbt/                # Star schema (fact_lap, fact_stint, fact_pit_stop)
├── app/                # Streamlit dashboard
├── api/                # FastAPI strategy endpoint (stretch)
├── docs/               # Methodology, model cards, writeups
├── tests/              # pytest + pandera schema tests
├── notebooks/          # EDA, model validation
└── powerbi/            # Executive .pbix view
```

---

## Resume bullets generated by this project

- Built an end-to-end Formula 1 race strategy analytics platform ingesting 4 seasons of lap-level timing and weather data (~90,000 lap rows across 85 races and 33 circuits) via FastF1 and Ergast APIs into a DuckDB + dbt warehouse with automated schema tests and GitHub Actions CI.
- Engineered tyre degradation models (isotonic regression per compound × circuit, 102 fitted curves) achieving 1.38s MAE on a 15k-lap within-circuit holdout, plus a bootstrap-CI pit-cost calculator that reproduces broadcast-knowledge values across all 33 circuits (Singapore 31.4s, Spa 20.5s).
- Trained a calibrated LightGBM undercut-success classifier (AUC 0.70, Brier 0.09) with isotonic probability calibration on 1,861 historical pit stops, deployed inside a 6-page Streamlit decision dashboard alongside a Monte Carlo race simulator.

---

## Acknowledgements

Built on [FastF1](https://github.com/theOehrly/Fast-F1), the [Ergast Developer API](http://ergast.com/mrd/), and [Open-Meteo](https://open-meteo.com/). Not affiliated with Formula 1, the FIA, or any team.

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Christian "Red" Callahan** — BI Analyst
[GitHub](https://github.com/CCallahan308) · [LinkedIn](https://www.linkedin.com/in/) · christian.g.callahan@gmail.com
