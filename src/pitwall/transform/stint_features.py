"""Stint-level feature engineering.

A "stint" is a contiguous run of laps on one set of tyres. This module turns
lap-level data into stint-level features used by the degradation model and
the undercut classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.config import CLEAN_AIR_GAP_S, IN_OUT_LAP_TRIM


def add_stint_position(laps: pd.DataFrame) -> pd.DataFrame:
    """Tag each lap with its 1-indexed position within its stint."""
    out = laps.sort_values(["Year", "Round", "Driver", "LapNumber"]).copy()
    out["StintPosition"] = out.groupby(["Year", "Round", "Driver", "Stint"]).cumcount() + 1
    return out


def is_clean_lap(laps: pd.DataFrame) -> pd.Series:
    """Boolean mask for laps suitable for degradation/pace modeling.

    Excludes: in-laps, out-laps, SC/VSC laps, deleted laps, laps with traffic
    (gap-ahead < threshold), and laps with no valid time.
    """
    mask = laps["LapTime"].notna()
    if "PitInTime" in laps:
        mask &= laps["PitInTime"].isna()
    if "PitOutTime" in laps:
        mask &= laps["PitOutTime"].isna()
    if "TrackStatus" in laps:
        # FastF1 uses single-char codes; '1' = clear track. Anything else = SC/VSC/yellow.
        mask &= laps["TrackStatus"].astype(str).str.fullmatch("1")
    if "Deleted" in laps:
        mask &= ~laps["Deleted"].fillna(False)
    if "GapAhead" in laps:
        mask &= laps["GapAhead"].fillna(99) >= CLEAN_AIR_GAP_S
    return mask


def stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, driver, stint) with deg slope and pace features."""
    df = add_stint_position(laps)
    df["CleanLap"] = is_clean_lap(df)

    rows = []
    for (year, rnd, drv, stint), g in df.groupby(["Year", "Round", "Driver", "Stint"]):
        clean = g[g["CleanLap"] & (g["StintPosition"] > IN_OUT_LAP_TRIM)]
        if len(clean) < 3:
            slope = np.nan
            intercept = np.nan
            r2 = np.nan
        else:
            x = clean["StintPosition"].to_numpy(dtype=float)
            y = clean["LapTimeFuelCorrected"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
            r2 = 1.0 - ss_res / ss_tot

        rows.append({
            "Year": int(year),
            "Round": int(rnd),
            "Driver": drv,
            "Stint": int(stint),
            "Compound": g["Compound"].iloc[0] if "Compound" in g else None,
            "StintLength": int(g["LapNumber"].nunique()),
            "FirstLap": int(g["LapNumber"].min()),
            "LastLap": int(g["LapNumber"].max()),
            "MeanPaceCleanS": float(clean["LapTimeFuelCorrected"].mean()) if len(clean) else np.nan,
            "MedianPaceCleanS": float(clean["LapTimeFuelCorrected"].median()) if len(clean) else np.nan,
            "DegSlopeSPerLap": float(slope) if not np.isnan(slope) else np.nan,
            "DegInterceptS": float(intercept) if not np.isnan(intercept) else np.nan,
            "DegFitR2": float(r2) if not np.isnan(r2) else np.nan,
            "NumCleanLaps": int(len(clean)),
        })
    return pd.DataFrame(rows)
