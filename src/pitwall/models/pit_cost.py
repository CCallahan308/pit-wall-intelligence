"""Pit-stop time loss model.

Pit loss = (in-lap time + out-lap time) - 2 * (driver's clean-air green-lap pace)

This is the canonical single number every F1 strategist tracks: the cost in
seconds of converting one green lap into a pit stop. Typical values:

  Monaco       ~21–23s   (slow pit lane, short lap)
  Monza        ~22–24s   (long pit lane, fast track)
  Singapore    ~26–28s   (slow pit lane, slow track)
  Spielberg    ~19–20s   (fastest pit lane in the calendar)

We compute this per circuit, with bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pit_loss_per_stop(
    laps: pd.DataFrame, pit_window: int = 3, clean_lap_quantile: float = 0.10
) -> pd.DataFrame:
    """One row per actual pit stop with the time lost vs. clean-air pace.

    Algorithm:
      1. find each lap where PitInTime is set → the "in-lap"
      2. the next lap is the "out-lap"
      3. baseline = `clean_lap_quantile`-th percentile of fuel-corrected lap
         times within ±`pit_window` laps of the stop, excluding in/out laps
      4. pit_loss = (in_lap + out_lap) - 2 * baseline
    """
    rows = []
    grouped = laps.sort_values(["Year", "Round", "Driver", "LapNumber"]).groupby(
        ["Year", "Round", "Driver"], sort=False
    )
    for (year, rnd, drv), g in grouped:
        g = g.reset_index(drop=True)
        in_lap_idx = g.index[g["PitInTime"].notna()].tolist() if "PitInTime" in g else []
        for i in in_lap_idx:
            if i + 1 >= len(g):
                continue
            in_lap = g.loc[i, "LapTimeFuelCorrected"]
            out_lap = g.loc[i + 1, "LapTimeFuelCorrected"]
            window = g.loc[max(0, i - pit_window): min(len(g) - 1, i + pit_window + 1)]
            clean = window.drop(index=[i, i + 1], errors="ignore")["LapTimeFuelCorrected"].dropna()
            if len(clean) < 2 or pd.isna(in_lap) or pd.isna(out_lap):
                continue
            baseline = clean.quantile(clean_lap_quantile)
            loss = (in_lap + out_lap) - 2 * baseline
            rows.append({
                "Year": int(year),
                "Round": int(rnd),
                "Driver": drv,
                "PitLap": int(g.loc[i, "LapNumber"]),
                "InLapS": float(in_lap),
                "OutLapS": float(out_lap),
                "BaselineS": float(baseline),
                "PitLossS": float(loss),
                "CircuitName": g.loc[i, "CircuitName"] if "CircuitName" in g else None,
            })
    return pd.DataFrame(rows)


def circuit_pit_cost(stops: pd.DataFrame, n_bootstrap: int = 1000) -> pd.DataFrame:
    """Median pit loss per circuit with bootstrap 95% CI."""
    rows = []
    rng = np.random.default_rng(42)
    for circuit, g in stops.groupby("CircuitName"):
        x = g["PitLossS"].dropna().to_numpy()
        if len(x) < 5:
            continue
        boot = rng.choice(x, size=(n_bootstrap, len(x)), replace=True).mean(axis=1)
        rows.append({
            "CircuitName": circuit,
            "MedianPitLossS": float(np.median(x)),
            "MeanPitLossS": float(np.mean(x)),
            "CI95LowS": float(np.percentile(boot, 2.5)),
            "CI95HighS": float(np.percentile(boot, 97.5)),
            "NStops": int(len(x)),
        })
    return pd.DataFrame(rows).sort_values("MedianPitLossS")
