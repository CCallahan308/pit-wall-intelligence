"""Tests for stint feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.transform.stint_features import add_stint_position, stint_summary


def _synthetic_laps(n: int = 20, deg_slope: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Year": [2024] * n,
            "Round": [1] * n,
            "Driver": ["VER"] * n,
            "Stint": [1] * n,
            "LapNumber": list(range(1, n + 1)),
            "Compound": ["MEDIUM"] * n,
            "LapTime": pd.to_timedelta(
                80 + deg_slope * np.arange(n) + rng.normal(0, 0.1, n), unit="s"
            ),
            "LapTimeFuelCorrected": 78 + deg_slope * np.arange(n) + rng.normal(0, 0.1, n),
            "TrackStatus": ["1"] * n,
            "PitInTime": [pd.NaT] * n,
            "PitOutTime": [pd.NaT] * n,
        }
    )


def test_stint_position_is_one_indexed_and_monotonic():
    df = _synthetic_laps()
    out = add_stint_position(df)
    assert out["StintPosition"].min() == 1
    assert (out["StintPosition"].diff().dropna() == 1).all()


def test_stint_summary_recovers_deg_slope():
    df = _synthetic_laps(n=30, deg_slope=0.08)
    summary = stint_summary(df)
    assert len(summary) == 1
    # The fitted slope should be within 0.02s/lap of the planted slope
    assert abs(summary["DegSlopeSPerLap"].iloc[0] - 0.08) < 0.02
