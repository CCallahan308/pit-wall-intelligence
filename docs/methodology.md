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

**Measured performance (85-race, 4-season sample):**
- Within-circuit MAE: **1.38 s** (random 20% stint holdout, 15,372 test laps)
- Cross-circuit MAE: **5.70 s** (entire Italian GP held out — improved from 9.58s on the 6-race sample as 32 reference circuits now anchor the compound-level fallback)

## 5. Pit stop cost

The canonical strategist's metric: how much time does it cost to convert one green lap into a pit stop?

```
pit_loss = (in_lap + out_lap) - 2 * clean_air_baseline
```

Where `clean_air_baseline` is the 10th percentile of fuel-corrected lap times within ±3 laps of the stop, excluding the stop itself.

Computed per circuit with bootstrap 95% CI. **Measured values (85-race, 33-circuit sample):**

| Circuit | Median | 95% CI | n stops |
|---|---|---|---|
| Belgian GP (Spa) | 20.46 s | [20.45, 21.69] | 74 |
| Miami GP | 20.94 s | [21.10, 23.44] | 33 |
| Australian GP | 21.79 s | [21.63, 24.65] | 33 |
| US GP | 22.50 s | [22.48, 23.66] | 98 |
| Austrian GP | 22.67 s | [23.11, 24.68] | 98 |
| Monaco GP | 23.04 s | [23.46, 26.82] | 37 |
| Bahrain GP | 25.85 s | [26.25, 27.24] | 154 |
| Italian GP (Monza) | 26.57 s | [26.93, 28.71] | 67 |
| British GP | 28.63 s | [27.08, 30.77] | 73 |
| Emilia Romagna GP (Imola) | 29.96 s | [30.74, 32.84] | 63 |
| Singapore GP | 31.42 s | [31.00, 32.73] | 25 |

33 circuits ingested in total — Singapore is the slowest pit lane, Spa the fastest. Reproducible via `scripts/train_and_validate.py`.

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

**Measured metrics (85-race sample, 1,861 stops → 1,395 train / 466 test):**
- AUC: **0.701**
- Brier score: **0.093**
- Base rate: 0.106 (positions gained within 5 laps of pit)

The model is calibrated via isotonic regression on 5-fold CV. The AUC is honest — it dropped from 0.74 on the small 6-race sample because larger data contains the harder, more typical pit decisions that don't fit a simple pattern.

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

## 8. Safety Car vs Virtual Safety Car distinction

A real strategy engineer never lumps SC and VSC pits with green-flag pits. The economics are different:

| Regime | FastF1 status code | What happens | Pit cost impact |
|---|---|---|---|
| **Green** | `1` | Normal racing | Full ~22-25s pit-loss penalty |
| **Yellow** | `2` | Local yellow flag | Modest pace impact, near-green |
| **SC** | `4` | Full Safety Car | Field bunches at SC pace; pit can cost as little as 3-5s |
| **Red** | `5` | Red-flag stoppage | Tyre changes often free (regulation-dependent) |
| **VSC** | `6` / `7` | Virtual Safety Car | Hold-delta-time rule — saves less than SC, sometimes costs *more* than green because the in-lap is slower |

`fact_pit_stop.pit_type` is derived by inspecting the track-status codes of the in-lap, out-lap, and ±2 surrounding laps. FastF1 packs multiple status transitions per lap into a single string (e.g. `'12'` = saw both green and yellow on the same lap), so we use substring presence rather than equality.

The classification order is **RED > SC > VSC > YELLOW > GREEN** — if any code in the window indicates a more-restrictive flag, that wins.

**Practical impact on the undercut classifier:** we train on GREEN stops only. Mixing in SC/VSC would let the model learn "the field gained positions when an SC was deployed" — a true but useless signal (you can't predict when an SC will arrive, and if one does the strategy decision is fundamentally different anyway).

## 9. What this is not

- **Not a race-winner predictor.** Winning a race is dominated by car performance, which we deliberately do not model. We quantify *decisions*, not *outcomes*.
- **Not real-time.** Live timing requires a paid feed; we use post-session FastF1 data.
- **Not team-private.** All inputs are publicly available; team-side strategy tools have access to far richer telemetry and tyre-energy data.

The goal is to demonstrate the *engineering and reasoning* a strategist applies, not to replicate their proprietary stack.
