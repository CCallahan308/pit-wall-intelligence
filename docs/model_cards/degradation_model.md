# Model Card — Tyre Degradation Curve

## Intended use

Predict the expected pace loss in seconds for a tyre at a given age, compound, and circuit, relative to a fresh tyre of the same compound at the same circuit.

## Model details

- **Algorithm:** Isotonic regression (`sklearn.isotonic.IsotonicRegression`)
- **Granularity:** one curve per `(compound, circuit)`; fallback to compound-only curve if < 30 samples
- **Confidence intervals:** bootstrap (1,000 resamples)

## Training data

**Current sample (6 races from 2024):**
- 5,388 clean lap rows across 6 circuits, 270 stints total
- 16 fitted per-(compound, circuit) curves + 3 compound-level fallbacks
- Inputs: stint position (tyre age in laps)
- Target: absolute fuel-corrected lap time (model now predicts absolute pace, not a delta — earlier
  delta-based design conflated training/test baselines)

## Performance (measured)

| Metric | Value | Notes |
|---|---|---|
| Within-circuit MAE | **0.834 s** | random 20% stint holdout (1,289 laps) |
| Cross-circuit MAE | **9.580 s** | Italian GP held out entirely (894 laps) — model has no circuit baseline |

The cross-circuit gap is the honest limitation: an isotonic curve per (compound, circuit) cannot
generalise to an unseen circuit without *some* practice data to calibrate the baseline. In production
this is fine — by race day teams always have FP1/FP2/FP3 stint data for the current circuit.

## Known limitations

- Tyre temperature warm-up is not modeled — first 2 laps of a stint are systematically underpredicted.
- No driver-specific tyre management residual yet — Hamilton and Pérez are softer on tyres than the global curve suggests.
- Cross-circuit generalisation requires per-circuit calibration data (see above).
- Wet/intermediate curves are sparse — many circuits have no wet samples.

## Known limitations

- Does not model tyre temperature warm-up — first 2 laps of a stint are systematically underpredicted.
- No driver-specific tyre management term — Hamilton and Pérez are notably softer on tyres than average; the global curve splits the difference.
- Wet/intermediate curves are sparse — many circuits have no wet samples.
