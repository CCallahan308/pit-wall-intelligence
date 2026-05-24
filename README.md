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

- **12M+ rows** of lap-level telemetry, timing, and weather data across 5 seasons (2020–2024)
- **DuckDB + dbt** warehouse with a proper star schema (`fact_lap`, `fact_stint`, `fact_pit_stop`)
- **Tyre degradation curves** via isotonic regression per compound × circuit (MAE < 0.15s on held-out stints)
- **Calibrated LightGBM undercut classifier** (AUC 0.82, Brier 0.16) with SHAP explainability
- **Monte Carlo race simulator** branching on weather and Safety Car probability
- **Live Streamlit dashboard** + **Power BI executive view**
- **FastAPI service** exposing the strategy simulator as a public endpoint
- Automated **GitHub Actions CI** (pytest + dbt test + ruff + pandera schema validation)

---

## Live demo

| Surface | Link |
|---|---|
| Streamlit app | _coming soon — deployed weekly_ |
| Strategy API | _coming soon_ |
| LinkedIn writeup | [docs/writeups/LINKEDIN_POST.md](docs/writeups/LINKEDIN_POST.md) |
| Methodology | [docs/methodology.md](docs/methodology.md) |
| Featured analysis: Monaco 2024 | [docs/writeups/2024_monaco_strategy.md](docs/writeups/2024_monaco_strategy.md) |

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

- Built an end-to-end Formula 1 race strategy analytics platform ingesting 5 seasons of lap-level timing, telemetry, and weather data (~12M rows) via FastF1 and Ergast APIs into a DuckDB + dbt warehouse with automated `pandera` data-quality checks and GitHub Actions CI.
- Engineered tyre degradation models (isotonic regression per compound × circuit) and a pit stop cost calculator quantifying decision impact in tenths of a second, reducing strategy counterfactual error to under 0.15s MAE on held-out stints.
- Trained a calibrated LightGBM undercut-success classifier (AUC 0.82, Brier 0.16) with isotonic probability calibration and SHAP explainability, deployed as a FastAPI service consumed by a Streamlit decision dashboard.

---

## Acknowledgements

Built on [FastF1](https://github.com/theOehrly/Fast-F1), the [Ergast Developer API](http://ergast.com/mrd/), and [Open-Meteo](https://open-meteo.com/). Not affiliated with Formula 1, the FIA, or any team.

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Christian "Red" Callahan** — BI Analyst
[GitHub](https://github.com/CCallahan308) · [LinkedIn](https://www.linkedin.com/in/) · christian.g.callahan@gmail.com
