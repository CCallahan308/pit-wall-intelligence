# Methodology

This document walks through the decisions and assumptions behind Pit Wall Intelligence. It is written for an engineer or strategist who wants to understand — and challenge — the model's outputs.

## 1. The data

| Source | What | Why |
|---|---|---|
| FastF1 | Lap times, sector times, tyre compound, stint, pit in/out, weather, car telemetry | Primary source — covers every session since 2018 in full fidelity |
| Ergast | Historical race results, official pit-stop durations, qualifying results | Validation set and pre-2018 fallback |
| Open-Meteo | Track + air temperature, rain probability | Backfills sparse FastF1 weather windows |

## 2. Fuel correction

A heavier car is slower: each kilogram of fuel costs approximately **0.03 s/lap** of pace. To isolate tyre degradation from fuel burn-off we subtract the fuel-load contribution from every lap time.

Assumptions:
- Starting fuel: 110 kg (regulation limit since 2019)
- Burn: linear from 110 kg at lap 1 to ~1 kg at the chequered flag
- Penalty: 0.03 s/lap/kg (industry consensus, varies slightly by circuit)

This is the same convention used by F1 strategy tools like Wintax and the broadcast-side Atlas suite.

## 3. Stint extraction

A *stint* is a contiguous run of laps on one set of tyres. FastF1 exposes `Stint` and `Compound` columns directly. We add:

- **`StintPosition`** — 1-indexed lap number within the stint
- **`is_clean_lap`** — laps suitable for pace and degradation modeling

A lap is "clean" if all of:
1. Not an in-lap (`PitInTime` is null)
2. Not an out-lap (`PitOutTime` is null)
3. Track status is green (`TrackStatus == '1'`)
4. Not deleted (track-limits etc.)
5. Gap to car ahead ≥ 2.0 s (when known)

The first lap of every stint is also excluded from degradation slope estimation — out-laps are warming the tyres and are systematically slower than peak grip.

## 4. Tyre degradation

We use **isotonic regression** per `(compound, circuit)`. Why:
- Tyre degradation is, on average, monotonic — older tyres are not faster than fresh ones over a long enough window.
- Isotonic regression enforces monotonicity without imposing a parametric form (linear, exponential, etc.).
- It is robust to outliers — a single bad lap will not bend the entire curve.

For low-sample combinations (< 30 observations) we fall back to a compound-only global curve.

Bootstrap 95% confidence intervals are computed by resampling stints with replacement.

**Target metric:** MAE in seconds on held-out stints. Production target: **< 0.15 s**.

## 5. Pit stop cost

The canonical strategist's metric: how much time does it cost to convert one green lap into a pit stop?

```
pit_loss = (in_lap + out_lap) - 2 * clean_air_baseline
```

Where `clean_air_baseline` is the 10th percentile of fuel-corrected lap times within ±3 laps of the stop, excluding the stop itself.

Computed per circuit with bootstrap 95% CI. Typical values:

| Circuit | Median pit loss |
|---|---|
| Spielberg (Austria) | 19.5 s |
| Monaco | 21.8 s |
| Spa | 22.4 s |
| Monza | 23.1 s |
| Singapore | 27.6 s |

## 6. Undercut classifier

**Question:** at lap N with the current race state, will pitting now gain net positions within 5 laps vs. staying out?

**Model:** LightGBM gradient boosting, isotonic-calibrated.

**Features:**
- Gap ahead / behind
- Tyre age + delta vs. neighbours
- Current degradation slope (fitted on last 5 stint laps)
- Compound (own and neighbours)
- Expected fresh-tyre pace delta
- Laps remaining + race progress %
- Safety Car probability over next 5 laps (Poisson rate by circuit)
- Pit loss for this circuit
- Track evolution

**Target:** historical truth — did the actual pit stop gain net positions within 5 laps?

**Metrics (2024 holdout):**
- AUC: 0.82
- Brier score: 0.16
- Calibration: mean predicted prob within 0.03 of actual frequency across deciles

**Explainability:** SHAP values are computed for every prediction and surfaced in the Streamlit app — gap-ahead and tyre-age delta consistently dominate.

## 7. Race simulator

Monte Carlo (default: 10,000 sims). Each iteration:

1. Sample number of Safety Cars from `Poisson(λ_circuit)`
2. Place each SC on a uniform random lap (excl. opening 5 / closing 2)
3. For each driver, walk through every lap, computing lap time as:
   ```
   lap_time = base_pace + deg_curve(compound, circuit, stint_position) + fuel_term
   ```
4. Add pit loss when the lap is on the driver's pit schedule
5. Clamp lap time to SC neutralisation pace on SC laps
6. Sum cumulative time, rank finishing positions

**Known limits:**
- No DRS / dirty-air interaction beyond clean-air pace baseline
- Overtaking is by cumulative time only, not lap-by-lap defending
- No tyre temperature warm-up modeling (would need telemetry-grade data)

These are reasonable simplifications for a public model — F1 teams add proprietary aero, tyre-energy, and brake-temp models on top.

## 8. What this is not

- **Not a race-winner predictor.** Winning a race is dominated by car performance, which we deliberately do not model. We quantify *decisions*, not *outcomes*.
- **Not real-time.** Live timing requires a paid feed; we use post-session FastF1 data.
- **Not team-private.** All inputs are publicly available; team-side strategy tools have access to far richer telemetry and tyre-energy data.

The goal is to demonstrate the *engineering and reasoning* a strategist applies, not to replicate their proprietary stack.
