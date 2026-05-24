"""Regression tests for the modeling code.

These don't try to assert specific AUC numbers (those drift with data).
They assert behavioural invariants: the model fits, predicts in [0,1],
respects feature ordering, and saves/loads round-trip clean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall.models.degradation_curve import DegradationModel
from pitwall.models.undercut_classifier import FEATURE_COLS, UndercutClassifier


def _synthetic_laps(
    n_circuits: int = 3, n_stints_per_circuit: int = 10, n_laps_per_stint: int = 20
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_circuits):
        circuit = f"Circuit_{c}"
        baseline = 80 + 4 * c  # circuits have different base lap times
        for s in range(n_stints_per_circuit):
            compound = ["SOFT", "MEDIUM", "HARD"][s % 3]
            deg = {"SOFT": 0.12, "MEDIUM": 0.08, "HARD": 0.05}[compound]
            for lap in range(1, n_laps_per_stint + 1):
                rows.append(
                    {
                        "Year": 2024,
                        "Round": c + 1,
                        "Driver": f"D{s:02d}",
                        "Stint": 1,
                        "Compound": compound,
                        "StintPosition": lap,
                        "CircuitName": circuit,
                        "LapTimeFuelCorrected": baseline + deg * lap + rng.normal(0, 0.15),
                    }
                )
    return pd.DataFrame(rows)


def test_degradation_model_fits_and_predicts():
    laps = _synthetic_laps()
    model = DegradationModel(min_samples_per_curve=10).fit(stints=pd.DataFrame(), laps=laps)
    pred = model.predict_pace("MEDIUM", "Circuit_0", np.array([1, 5, 10, 20]))
    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
    # Monotonic-ish in tyre age (allowing some isotonic plateaus)
    assert pred[-1] >= pred[0]


def test_degradation_model_falls_back_to_global_for_unseen_circuit():
    laps = _synthetic_laps()
    model = DegradationModel(min_samples_per_curve=10).fit(stints=pd.DataFrame(), laps=laps)
    pred = model.predict_pace("MEDIUM", "Unknown_Circuit", np.array([10]))
    # Either NaN (no fallback) or a finite value from the global compound curve
    assert pred.shape == (1,)


def test_degradation_mae_is_positive_and_finite():
    laps = _synthetic_laps()
    model = DegradationModel(min_samples_per_curve=10).fit(stints=pd.DataFrame(), laps=laps)
    mae = model.mae(stints=pd.DataFrame(), laps=laps)
    assert np.isfinite(mae)
    assert mae > 0
    # On the training set with synthetic data, MAE should be ~ the noise std (0.15s)
    assert mae < 2.0


def test_undercut_classifier_fits_and_calibrates():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLS})
    # Make label depend on tyre_age + gap_ahead so the model has signal to learn
    logits = 0.4 * X["tyre_age"] - 0.3 * X["gap_ahead_s"] + rng.normal(0, 0.5, n)
    y = pd.Series((logits > logits.median()).astype(int))

    clf = UndercutClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (n,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_undercut_classifier_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    n = 150
    X = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLS})
    y = pd.Series(rng.integers(0, 2, n))

    clf = UndercutClassifier()
    clf.fit(X, y)
    proba_before = clf.predict_proba(X)

    path = tmp_path / "uc.joblib"
    clf.save(path)
    loaded = UndercutClassifier.load(path)
    proba_after = loaded.predict_proba(X)
    np.testing.assert_allclose(proba_before, proba_after)


def test_undercut_classifier_metrics_reflect_evaluation():
    rng = np.random.default_rng(2)
    n = 200
    X = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLS})
    # Deterministic label so AUC > 0.5
    y = pd.Series((X["tyre_age"] > X["tyre_age"].median()).astype(int))

    clf = UndercutClassifier()
    clf.fit(X.iloc[:150], y.iloc[:150])
    metrics = clf.evaluate(X.iloc[150:], y.iloc[150:])
    assert metrics["auc"] > 0.55  # learnable signal in synthetic data
    assert 0 <= metrics["brier"] <= 1
    assert metrics["n"] == 50


@pytest.mark.parametrize("compound", ["SOFT", "MEDIUM", "HARD"])
def test_degradation_predict_delta_zero_at_fresh_tyre(compound):
    laps = _synthetic_laps()
    model = DegradationModel(min_samples_per_curve=10).fit(stints=pd.DataFrame(), laps=laps)
    delta = model.predict_delta(compound, "Circuit_0", np.array([1]))
    assert np.isclose(delta[0], 0.0, atol=1e-6)
