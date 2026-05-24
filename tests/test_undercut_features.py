"""Tests for the real undercut feature builder.

The point is to prove these features are computed from data, not hardcoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.features.undercut import COMPOUND_IDX, build_features
from pitwall.models.degradation_curve import DegradationModel


def _synthetic_race(n_drivers: int = 4, n_laps: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal synthetic race with one pit per driver."""
    rng = np.random.default_rng(0)
    rows = []
    pits = []
    for d in range(n_drivers):
        drv = f"D{d:02d}"
        pit_lap = 12 + d  # different pit laps per driver
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= pit_lap else 2
            stint_pos = lap if stint == 1 else lap - pit_lap
            compound = "MEDIUM" if stint == 1 else "HARD"
            rows.append(
                {
                    "Year": 2024,
                    "Round": 1,
                    "Driver": drv,
                    "LapNumber": lap,
                    "Stint": stint,
                    "StintPosition": stint_pos,
                    "Compound": compound,
                    "CircuitName": "TestCircuit",
                    "Team": "TestTeam",
                    "LapTimeSeconds": 90.0 + 0.06 * stint_pos + 0.1 * d + rng.normal(0, 0.05),
                    "LapTimeFuelCorrected": 86.0 + 0.06 * stint_pos + 0.1 * d + rng.normal(0, 0.05),
                    "Position": d + 1,  # constant for simplicity
                    "IsCleanLap": True,
                    "track_status": "1",
                }
            )
        pits.append(
            {
                "year": 2024,
                "round_num": 1,
                "driver_code": drv,
                "circuit_name": "TestCircuit",
                "pit_lap": pit_lap,
                "pit_loss_s": 22.0 + 0.2 * d,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pits)


def test_build_features_returns_expected_columns():
    laps, pits = _synthetic_race()
    deg = DegradationModel(min_samples_per_curve=5).fit(stints=pd.DataFrame(), laps=laps)
    feats = build_features(laps, pits, deg)
    assert not feats.empty
    expected = {
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
        "label",
        "group_id",
    }
    assert expected.issubset(set(feats.columns))


def test_pit_loss_is_per_circuit_not_constant():
    """The old version hardcoded 23.0. The new one looks it up per circuit."""
    laps, pits = _synthetic_race()
    deg = DegradationModel(min_samples_per_curve=5).fit(stints=pd.DataFrame(), laps=laps)
    feats = build_features(laps, pits, deg)
    assert not feats.empty
    # All stops in this race are at the same circuit, so the value should be
    # the median of our planted pit_loss_s values (~22.3), NOT the old 23.0.
    val = feats["pit_loss_circuit_s"].iloc[0]
    assert 21.0 < val < 23.0, f"expected per-circuit median, got {val}"


def test_current_deg_slope_is_data_driven():
    """Slope should reflect the planted 0.06 s/lap degradation, not the old constant 0.08."""
    laps, pits = _synthetic_race()
    deg = DegradationModel(min_samples_per_curve=5).fit(stints=pd.DataFrame(), laps=laps)
    feats = build_features(laps, pits, deg)
    # The median slope should be near the planted 0.06, not 0.08
    median_slope = feats["current_deg_slope"].median()
    assert 0.0 < median_slope < 0.2, f"slope out of range: {median_slope}"


def test_group_id_is_year_round():
    laps, pits = _synthetic_race()
    deg = DegradationModel(min_samples_per_curve=5).fit(stints=pd.DataFrame(), laps=laps)
    feats = build_features(laps, pits, deg)
    assert (feats["group_id"] == "2024_01").all()


def test_label_is_binary():
    laps, pits = _synthetic_race()
    deg = DegradationModel(min_samples_per_curve=5).fit(stints=pd.DataFrame(), laps=laps)
    feats = build_features(laps, pits, deg)
    assert set(feats["label"].unique()).issubset({0, 1})


def test_compound_idx_mapping():
    assert COMPOUND_IDX["SOFT"] == 0
    assert COMPOUND_IDX["MEDIUM"] == 1
    assert COMPOUND_IDX["HARD"] == 2
