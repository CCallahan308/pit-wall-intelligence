"""Tests for fuel correction — the linchpin of degradation modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.transform.fuel_correction import (
    apply,
    fuel_corrected_time,
    fuel_mass_at_lap,
)


def test_fuel_mass_decreases_linearly():
    masses = fuel_mass_at_lap(np.array([1, 2, 50]), total_laps=58)
    assert masses[0] > masses[1] > masses[2]
    assert masses[0] == 110.0  # starts at full fuel


def test_fuel_mass_never_negative():
    masses = fuel_mass_at_lap(np.arange(1, 100), total_laps=58)
    assert (masses >= 0).all()


def test_fuel_correction_subtracts_load():
    raw = pd.Series([90.0, 90.0, 90.0])
    laps = pd.Series([1, 2, 3])
    corrected = fuel_corrected_time(raw, laps, total_laps=58)
    # Lap 1 has more fuel, so its corrected time should be lower (faster)
    assert corrected.iloc[0] < corrected.iloc[1] < corrected.iloc[2]


def test_apply_adds_column():
    df = pd.DataFrame({
        "Year": [2024] * 4,
        "Round": [1] * 4,
        "Driver": ["VER"] * 4,
        "LapNumber": [1, 2, 3, 4],
        "LapTime": [90.0, 89.8, 89.6, 89.4],
    })
    out = apply(df)
    assert "LapTimeFuelCorrected" in out.columns
    assert out["LapTimeFuelCorrected"].notna().all()
    # Corrected times should be lower than raw (we subtracted fuel mass)
    assert (out["LapTimeFuelCorrected"] < out["LapTimeSeconds"]).all()
