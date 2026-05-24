"""Undercut success classifier.

Question: at the current race state, will pitting NOW gain net track position
within the next 5 laps vs. staying out?

Model:
  - LightGBM gradient boosting
  - Isotonic calibration on a held-out slice for honest probabilities
  - SHAP global + per-prediction explanations (used in the Streamlit app)

Features (all engineered from stint + lap data):
  - gap_ahead_s, gap_behind_s
  - tyre_age, tyre_age_delta_vs_ahead, tyre_age_delta_vs_behind
  - compound, compound_ahead, compound_behind
  - current_deg_slope, expected_fresh_pace_delta
  - laps_remaining, race_progress_pct
  - sc_prob_next_5  (Poisson rate by circuit)
  - pit_loss_circuit_s
  - track_evolution_s_per_lap

Target: did the driver gain net positions within 5 laps post-pit, on actual
historical stops? Negative class = stops that LOST positions; positive class
= stops that GAINED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score


FEATURE_COLS: list[str] = [
    "gap_ahead_s",
    "gap_behind_s",
    "tyre_age",
    "tyre_age_delta_vs_ahead",
    "tyre_age_delta_vs_behind",
    "compound_idx",
    "current_deg_slope",
    "expected_fresh_pace_delta",
    "laps_remaining",
    "race_progress_pct",
    "sc_prob_next_5",
    "pit_loss_circuit_s",
    "track_evolution_s_per_lap",
]


@dataclass
class UndercutClassifier:
    model: CalibratedClassifierCV | None = None
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    metrics: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UndercutClassifier":
        base = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=5)
        self.model.fit(X[self.feature_cols], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() the model first"
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        proba = self.predict_proba(X)
        auc = roc_auc_score(y, proba)
        brier = brier_score_loss(y, proba)
        self.metrics = {"auc": float(auc), "brier": float(brier), "n": int(len(y))}
        return self.metrics

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: str | Path) -> "UndercutClassifier":
        import joblib

        blob = joblib.load(path)
        obj = cls(model=blob["model"], feature_cols=blob["feature_cols"], metrics=blob.get("metrics", {}))
        return obj
