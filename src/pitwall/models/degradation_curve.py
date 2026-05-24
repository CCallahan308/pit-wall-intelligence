"""Tyre degradation curves via isotonic regression.

Why isotonic? Tyre degradation is, on average, monotonic — older tyres are
not faster than fresh ones over a long enough window. Isotonic regression
enforces this constraint while remaining non-parametric (no functional form
imposed), which fits the underlying physics better than a polynomial.

Fits one curve per (compound, circuit). For low-sample circuit/compound
combinations we fall back to a compound-only global curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class DegradationModel:
    """Holds isotonic curves keyed by (compound, circuit)."""

    curves: dict[tuple[str, str], IsotonicRegression] = field(default_factory=dict)
    global_curves: dict[str, IsotonicRegression] = field(default_factory=dict)
    min_samples_per_curve: int = 30

    def fit(self, stints: pd.DataFrame, laps: pd.DataFrame) -> "DegradationModel":
        """Fit one curve per compound × circuit.

        stints : output of `transform.stint_features.stint_summary`
        laps   : laps DataFrame with LapTimeFuelCorrected, Compound, CircuitName, StintPosition
        """
        df = laps.merge(stints[["Year", "Round", "Driver", "Stint", "DegFitR2"]],
                        on=["Year", "Round", "Driver", "Stint"], how="left")
        df = df.dropna(subset=["LapTimeFuelCorrected", "Compound", "StintPosition", "CircuitName"])

        # Global per-compound fallback first
        for compound, g in df.groupby("Compound"):
            model = IsotonicRegression(out_of_bounds="clip")
            baseline = g["LapTimeFuelCorrected"].quantile(0.05)
            y = g["LapTimeFuelCorrected"] - baseline
            model.fit(g["StintPosition"].to_numpy(), y.to_numpy())
            self.global_curves[compound] = model

        for (compound, circuit), g in df.groupby(["Compound", "CircuitName"]):
            if len(g) < self.min_samples_per_curve:
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            baseline = g["LapTimeFuelCorrected"].quantile(0.05)
            y = g["LapTimeFuelCorrected"] - baseline
            model.fit(g["StintPosition"].to_numpy(), y.to_numpy())
            self.curves[(compound, circuit)] = model

        return self

    def predict_delta(self, compound: str, circuit: str, tyre_age: np.ndarray | int) -> np.ndarray:
        """Predicted *added* seconds vs. a fresh tyre at the same circuit."""
        ages = np.atleast_1d(np.asarray(tyre_age, dtype=float))
        key = (compound, circuit)
        if key in self.curves:
            return self.curves[key].predict(ages)
        if compound in self.global_curves:
            return self.global_curves[compound].predict(ages)
        return np.full_like(ages, np.nan)

    def mae(self, stints: pd.DataFrame, laps: pd.DataFrame) -> float:
        """Mean absolute error in seconds on the supplied laps."""
        df = laps.dropna(subset=["LapTimeFuelCorrected", "Compound", "StintPosition", "CircuitName"]).copy()
        preds = []
        for (compound, circuit), g in df.groupby(["Compound", "CircuitName"]):
            pred = self.predict_delta(compound, circuit, g["StintPosition"].to_numpy())
            baseline = g["LapTimeFuelCorrected"].quantile(0.05)
            preds.append(pd.Series(pred + baseline, index=g.index, name="pred"))
        if not preds:
            return float("nan")
        pred_series = pd.concat(preds).reindex(df.index)
        return float(np.nanmean(np.abs(df["LapTimeFuelCorrected"] - pred_series)))
