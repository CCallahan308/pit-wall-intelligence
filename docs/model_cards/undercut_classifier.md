# Model Card — Undercut Success Classifier

## Intended use

Given the current race state at lap N, estimate the probability that pitting now will result in net positions gained within the next 5 laps, vs. staying out.

**Intended consumers:** race strategists, broadcasters, fantasy/betting analytics, fans.
**Out of scope:** predicting race winners, predicting incidents, pace beyond a 5-lap horizon.

## Model details

- **Algorithm:** LightGBM gradient boosting (400 trees, lr=0.05, num_leaves=31)
- **Calibration:** Isotonic, 5-fold CV via `CalibratedClassifierCV`
- **Library:** `lightgbm==4.3`, `scikit-learn==1.4`

## Training data

**Current sample (6 races from 2024):**
- 123 historical pit stops total, 9% positive base rate (positions gained within 5 laps)
- 92 training / 31 test (75/25 random split, seed=42)

The labels come from real position deltas; many features (gap_ahead, gap_behind, deg_slope) are
currently filled with reasonable placeholders because lap-by-lap gap data requires the Ergast lap
endpoint, which the active environment cannot reach. Wiring those in is the next step.

## Performance (measured, not aspirational)

| Metric | Value | Notes |
|---|---|---|
| AUC | **0.741** | on 31-stop test set |
| Brier score | **0.071** | calibrated probabilities (isotonic, 5-fold CV) |
| Base rate | 0.089 | class imbalance — positive examples are rare in a 6-race sample |

These numbers are honest but small-N. At full-season scale (~600 stops) we expect AUC to stabilise
and the base rate to grow as we capture more competitive pit-window decisions vs. forced/SC stops.

## Feature importance

Not yet computed — SHAP is wired through `[explain]` extras (`uv sync --extra explain`) but
requires a re-run after the full season is loaded for the global ranking to be meaningful.

## Known limitations

- **Survivor bias:** we only train on historical stops that actually happened. Counterfactual stops (the ones strategists *didn't* take) are absent from the training distribution.
- **No team-strategy context:** the model does not know what the team plans next.
- **Weather sensitivity:** when rain probability is high, the model's confidence drops; flagged in the UI.
- **2022 regulation change:** ground-effect cars degrade differently. Models trained pre-2022 should be retrained.

## Ethical considerations

This model uses only publicly available data and is not a tool for unauthorized access to team-proprietary systems. Outputs are advisory and explicitly probabilistic.
