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

**85 races across 4 seasons (2020, 2021, 2023, 2024):**
- 1,861 historical green-flag pit stops total, 10.6% positive base rate (positions gained within 5 laps)
- 1,395 training / 466 test (75/25 random split, seed=42)

Labels come from real position deltas in `fact_lap`. Several features (gap_ahead, gap_behind, current_deg_slope) are currently filled with reasonable constants because lap-by-lap gap data requires the Ergast lap endpoint and per-stint slope estimation. Adding those is the highest-leverage next improvement.

## Performance (measured)

| Metric | Value | Notes |
|---|---|---|
| AUC | **0.701** | on 466-stop test set |
| Brier score | **0.093** | calibrated probabilities (isotonic, 5-fold CV) |
| Base rate | 0.106 | balanced enough for meaningful calibration |

The AUC is slightly lower than the 6-race sample (0.74) because the larger sample includes the harder, more typical pit decisions; the easy positives from a tiny sample don't dominate any more. This is the more honest production number.

## Feature importance

SHAP is wired through `[explain]` extras (`uv sync --extra explain`). Next pass will add a global ranking plot to this card.

## Known limitations

- **Survivor bias:** we only train on historical stops that actually happened. Counterfactual stops (the ones strategists *didn't* take) are absent from the training distribution.
- **No team-strategy context:** the model does not know what the team plans next.
- **Weather sensitivity:** when rain probability is high, the model's confidence drops; flagged in the UI.
- **2022 regulation change:** ground-effect cars degrade differently. Models trained pre-2022 should be retrained.

## Ethical considerations

This model uses only publicly available data and is not a tool for unauthorized access to team-proprietary systems. Outputs are advisory and explicitly probabilistic.
