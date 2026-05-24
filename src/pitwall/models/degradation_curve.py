"""Tyre degradation curves via isotonic regression.

Why isotonic? Tyre degradation is, on average, monotonic - older tyres are
not faster than fresh ones over a long enough window. Isotonic regression
enforces this constraint while remaining non-parametric (no functional form
imposed), which fits the underlying physics better than a polynomial.

The model fits *absolute fuel-corrected lap time* as a function of tyre age,
keyed on (compound, circuit). A circuit-specific intercept is stored so the
model can return either:

  - `predict_pace`  : absolute predicted lap time at this circuit & compound
  - `predict_delta` : pace loss vs. a fresh tyre at the same circuit

For low-sample (compound, circuit) combinations we fall back to a
compound-only curve fit on cross-circuit residuals (each circuit's lap times
centred on its own minimum-tyre-age median).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class DegradationModel:
    curves: dict[tuple[str, str], IsotonicRegression] = field(default_factory=dict)
    global_curves: dict[str, IsotonicRegression] = field(default_factory=dict)
    # Circuit-specific baseline pace (median of fastest 10% per circuit) -
    # used to translate global (delta) curves into absolute predictions.
    circuit_baseline: dict[str, float] = field(default_factory=dict)
    min_samples_per_curve: int = 30

    def fit(self, stints: pd.DataFrame, laps: pd.DataFrame) -> DegradationModel:
        df = laps.dropna(
            subset=["LapTimeFuelCorrected", "Compound", "StintPosition", "CircuitName"]
        )

        # Per-circuit baseline = 10th-percentile fuel-corrected pace at that circuit.
        # Used as the anchor when the global compound curve is applied.
        for circuit, g in df.groupby("CircuitName"):
            self.circuit_baseline[circuit] = float(g["LapTimeFuelCorrected"].quantile(0.10))

        # Global per-compound curve fit on *residuals* (lap_time - circuit_baseline)
        # so the curve generalises across circuits.
        for compound, g in df.groupby("Compound"):
            residual = (
                g["LapTimeFuelCorrected"].to_numpy()
                - g["CircuitName"].map(self.circuit_baseline).to_numpy()
            )
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(g["StintPosition"].to_numpy(), residual)
            self.global_curves[compound] = model

        # Per-(compound, circuit) curve fit on absolute lap times.
        for (compound, circuit), g in df.groupby(["Compound", "CircuitName"]):
            if len(g) < self.min_samples_per_curve:
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(g["StintPosition"].to_numpy(), g["LapTimeFuelCorrected"].to_numpy())
            self.curves[(compound, circuit)] = model

        return self

    def predict_pace(self, compound: str, circuit: str, tyre_age: np.ndarray | int) -> np.ndarray:
        """Absolute predicted fuel-corrected lap time."""
        ages = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        key = (compound, circuit)
        if key in self.curves:
            return self.curves[key].predict(ages)
        if compound in self.global_curves:
            baseline = self.circuit_baseline.get(
                circuit, np.nanmedian(list(self.circuit_baseline.values()))
            )
            return self.global_curves[compound].predict(ages) + baseline
        return np.full_like(ages, np.nan)

    def predict_delta(self, compound: str, circuit: str, tyre_age: np.ndarray | int) -> np.ndarray:
        """Pace loss vs. a fresh tyre of the same compound at the same circuit."""
        ages = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        fresh = self.predict_pace(compound, circuit, np.array([1.0]))
        return self.predict_pace(compound, circuit, ages) - fresh

    def mae(self, stints: pd.DataFrame, laps: pd.DataFrame) -> float:
        """MAE in seconds between predicted and actual fuel-corrected lap time."""
        df = laps.dropna(
            subset=["LapTimeFuelCorrected", "Compound", "StintPosition", "CircuitName"]
        ).copy()
        preds = np.empty(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            preds[i] = self.predict_pace(row.Compound, row.CircuitName, row.StintPosition)[0]
        actual = df["LapTimeFuelCorrected"].to_numpy()
        return float(np.nanmean(np.abs(actual - preds)))
