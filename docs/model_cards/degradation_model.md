# Model Card — Tyre Degradation Curve

## Intended use

Predict the expected pace loss in seconds for a tyre at a given age, compound, and circuit, relative to a fresh tyre of the same compound at the same circuit.

## Model details

- **Algorithm:** Isotonic regression (`sklearn.isotonic.IsotonicRegression`)
- **Granularity:** one curve per `(compound, circuit)`; fallback to compound-only curve if < 30 samples
- **Confidence intervals:** bootstrap (1,000 resamples)

## Training data

- 2020–2024 race sessions (~12 M lap rows, ~3,200 stints used after clean-lap filtering)
- Inputs: `stint_position` (tyre age in laps), fuel-corrected lap time
- Targets: fuel-corrected lap time minus 5th-percentile baseline

## Performance

| Metric | Value | Target |
|---|---|---|
| MAE on held-out stints | 0.13 s | < 0.15 s |
| Bias (mean residual) | +0.01 s | \|x\| < 0.05 s |

## Known limitations

- Does not model tyre temperature warm-up — first 2 laps of a stint are systematically underpredicted.
- No driver-specific tyre management term — Hamilton and Pérez are notably softer on tyres than average; the global curve splits the difference.
- Wet/intermediate curves are sparse — many circuits have no wet samples.
