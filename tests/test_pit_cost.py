"""Tests for the pit-loss model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.models.pit_cost import circuit_pit_cost, pit_loss_per_stop


def _build_race_with_stop():
    n = 30
    df = pd.DataFrame({
        "Year": [2024] * n,
        "Round": [1] * n,
        "Driver": ["VER"] * n,
        "LapNumber": list(range(1, n + 1)),
        "LapTimeFuelCorrected": [80.0] * n,
        "PitInTime": [pd.NaT] * n,
        "PitOutTime": [pd.NaT] * n,
        "CircuitName": ["Monaco"] * n,
    })
    # Plant a stop on lap 15 → in-lap +22s, out-lap +2s
    df.loc[14, "PitInTime"] = pd.Timestamp("2024-01-01")
    df.loc[14, "LapTimeFuelCorrected"] = 102.0
    df.loc[15, "LapTimeFuelCorrected"] = 82.0
    return df


def test_pit_loss_recovers_planted_value():
    df = _build_race_with_stop()
    stops = pit_loss_per_stop(df)
    assert len(stops) == 1
    # planted: (102 + 82) - 2*80 = 24s
    assert abs(stops["PitLossS"].iloc[0] - 24.0) < 0.5


def test_circuit_pit_cost_returns_ci():
    # Repeat the planted stop across drivers to build a sample
    races = []
    for drv in ["VER", "HAM", "LEC", "NOR", "RUS", "SAI", "PER"]:
        df = _build_race_with_stop()
        df["Driver"] = drv
        races.append(df)
    big = pd.concat(races, ignore_index=True)
    stops = pit_loss_per_stop(big)
    cost = circuit_pit_cost(stops, n_bootstrap=200)
    assert len(cost) == 1
    assert cost["CI95LowS"].iloc[0] < cost["MedianPitLossS"].iloc[0] < cost["CI95HighS"].iloc[0]
