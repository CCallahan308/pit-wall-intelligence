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

- 2020–2023 race sessions (4 seasons, ~88 races)
- Each historical pit stop becomes one positive or negative example based on the 5-lap-forward position change
- Training set: ~3,400 stops; held-out 2024 season: ~860 stops

## Performance

| Metric | Value (2024 holdout) | Notes |
|---|---|---|
| AUC | 0.82 | |
| Brier score | 0.16 | |
| Log loss | 0.51 | |
| Calibration error (ECE) | 0.029 | mean abs diff between predicted prob and actual freq across 10 bins |

## Feature importance (top 5, SHAP global)

1. `tyre_age_delta_vs_ahead`
2. `gap_ahead_s`
3. `current_deg_slope`
4. `laps_remaining`
5. `expected_fresh_pace_delta`

## Known limitations

- **Survivor bias:** we only train on historical stops that actually happened. Counterfactual stops (the ones strategists *didn't* take) are absent from the training distribution.
- **No team-strategy context:** the model does not know what the team plans next.
- **Weather sensitivity:** when rain probability is high, the model's confidence drops; flagged in the UI.
- **2022 regulation change:** ground-effect cars degrade differently. Models trained pre-2022 should be retrained.

## Ethical considerations

This model uses only publicly available data and is not a tool for unauthorized access to team-proprietary systems. Outputs are advisory and explicitly probabilistic.
