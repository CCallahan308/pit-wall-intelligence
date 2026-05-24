"""Fuel-load correction for lap times.

A heavier car is slower: each kg of fuel costs roughly 0.03 s/lap. To isolate
tyre degradation from fuel burn-off we subtract the fuel-load contribution,
so all laps are reported as if the car had crossed the line at minimum fuel.

This is the same convention used by F1 strategy software (e.g. Wintax, Atlas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.config import (
    FUEL_BURN_KG_PER_LAP,
    FUEL_TIME_PENALTY_S_PER_KG,
    STARTING_FUEL_KG,
)


def fuel_mass_at_lap(lap_number: pd.Series | np.ndarray, total_laps: int | None = None) -> np.ndarray:
    """Estimate kg of fuel remaining at the start of `lap_number`.

    Assumes linear burn from STARTING_FUEL_KG at lap 1 to ~1kg reserve at the
    final lap. Falls back to FUEL_BURN_KG_PER_LAP if total_laps is unknown.
    """
    lap = np.asarray(lap_number, dtype=float)
    if total_laps is None or total_laps <= 1:
        return np.maximum(STARTING_FUEL_KG - FUEL_BURN_KG_PER_LAP * (lap - 1), 0.0)
    burn = (STARTING_FUEL_KG - 1.0) / (total_laps - 1)
    return np.maximum(STARTING_FUEL_KG - burn * (lap - 1), 0.0)


def fuel_corrected_time(
    lap_time_s: pd.Series, lap_number: pd.Series, total_laps: int | None = None
) -> pd.Series:
    """Return lap time adjusted as if the car were on minimum fuel.

    fuel_corrected = raw - mass_kg * penalty_s_per_kg
    """
    mass = fuel_mass_at_lap(lap_number.to_numpy(), total_laps)
    return lap_time_s - mass * FUEL_TIME_PENALTY_S_PER_KG


def apply(laps: pd.DataFrame, group_cols: tuple[str, ...] = ("Year", "Round", "Driver")) -> pd.DataFrame:
    """Add a `LapTimeFuelCorrected` column to a laps dataframe.

    Expects columns: LapTime (timedelta or float seconds), LapNumber, plus
    group_cols identifying a unique race-driver combination.
    """
    out = laps.copy()
    if pd.api.types.is_timedelta64_dtype(out["LapTime"]):
        out["LapTimeSeconds"] = out["LapTime"].dt.total_seconds()
    else:
        out["LapTimeSeconds"] = out["LapTime"]

    out["LapTimeFuelCorrected"] = np.nan
    for _, idx in out.groupby(list(group_cols)).groups.items():
        sub = out.loc[idx]
        total = int(sub["LapNumber"].max())
        out.loc[idx, "LapTimeFuelCorrected"] = fuel_corrected_time(
            sub["LapTimeSeconds"], sub["LapNumber"], total_laps=total
        ).to_numpy()
    return out
