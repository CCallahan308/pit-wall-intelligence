"""Real feature engineering for the undercut classifier.

Replaces the earlier scaffold where 7/13 features were hardcoded constants.
Every feature here is derived from `fact_lap` and `fact_pit_stop`.

Features built:
  - gap_ahead_s, gap_behind_s              cumulative-race-time gap to the
                                            car ahead/behind at the in-lap
  - tyre_age, tyre_age_delta_vs_ahead/_vs_behind   StintPosition deltas
  - compound_idx                           ordinal compound code
  - current_deg_slope                      OLS slope over the last 5 clean
                                            laps of the current stint
  - expected_fresh_pace_delta              degradation model's pace loss at
                                            the current tyre age
  - laps_remaining, race_progress_pct      race-clock features
  - sc_prob_next_5                         empirical SC arrival rate over
                                            the next 5 laps for this circuit
  - pit_loss_circuit_s                     measured median pit loss for this
                                            circuit (was hardcoded 23.0s)
  - track_evolution_s_per_lap              median lap-over-lap pace gain
                                            across drivers in clean conditions
                                            for this race
  - label                                  +1 if track position improved
                                            5 laps after pit, else 0
  - group_id                               `(year, round_num)` for grouped CV

The output is a tidy frame ready for `GroupKFold`-style validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.models.degradation_curve import DegradationModel

COMPOUND_IDX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


@dataclass
class FeatureContext:
    """Pre-computed per-race lookups shared across stops."""

    cum_time_by_lap: dict[tuple[int, int, str], float]
    position_at_lap: dict[tuple[int, int, str, int], int]
    stint_position_at_lap: dict[tuple[int, int, str, int], int]
    pit_loss_by_circuit: dict[str, float]
    sc_rate_by_circuit: dict[str, float]
    track_evo_by_race: dict[tuple[int, int], float]
    total_laps_by_race: dict[tuple[int, int], int]


def _build_context(all_laps: pd.DataFrame, pits: pd.DataFrame) -> FeatureContext:
    """Pre-compute per-race lookups that are reused for every pit stop."""

    cum_time: dict[tuple[int, int, str], float] = {}
    position_at_lap: dict[tuple[int, int, str, int], int] = {}
    stint_position_at_lap: dict[tuple[int, int, str, int], int] = {}

    # Sort once for cumsum
    df = all_laps.sort_values(["Year", "Round", "Driver", "LapNumber"]).copy()
    df["cum_race_time_s"] = df.groupby(["Year", "Round", "Driver"])["LapTimeSeconds"].cumsum()

    for row in df.itertuples(index=False):
        key = (row.Year, row.Round, row.Driver, row.LapNumber)
        if pd.notna(row.cum_race_time_s):
            cum_time[(row.Year, row.Round, row.Driver, row.LapNumber)] = float(row.cum_race_time_s)
        if pd.notna(row.Position):
            position_at_lap[key] = int(row.Position)
        if pd.notna(row.StintPosition):
            stint_position_at_lap[key] = int(row.StintPosition)

    # Pit loss per circuit
    pit_loss_by_circuit: dict[str, float] = {}
    if not pits.empty:
        for circuit, g in pits.groupby("circuit_name"):
            pit_loss_by_circuit[circuit] = float(g["pit_loss_s"].median())

    # SC arrival rate per circuit — fraction of laps under non-green status
    sc_rate_by_circuit: dict[str, float] = {}
    for circuit, g in all_laps.groupby("CircuitName"):
        if "track_status" not in g.columns:
            sc_rate_by_circuit[circuit] = 0.05
            continue
        sc_share = float((g["track_status"].astype(str) != "1").mean())
        sc_rate_by_circuit[circuit] = sc_share

    # Track evolution: median lap-over-lap pace gain across drivers in the
    # first quarter of the race (when fuel is high and tyres are fresh).
    track_evo_by_race: dict[tuple[int, int], float] = {}
    for (year, rnd), g in all_laps.groupby(["Year", "Round"]):
        clean = g[g.get("IsCleanLap", True) & (g["LapNumber"] <= g["LapNumber"].max() // 4)]
        if len(clean) < 20:
            track_evo_by_race[(year, rnd)] = -0.02
            continue
        slopes = []
        for _, drv_laps in clean.groupby("Driver"):
            if len(drv_laps) < 5:
                continue
            x = drv_laps["LapNumber"].to_numpy(dtype=float)
            y = drv_laps["LapTimeFuelCorrected"].to_numpy(dtype=float)
            if np.isnan(y).all():
                continue
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                continue
            slope, _ = np.polyfit(x[mask], y[mask], 1)
            slopes.append(slope)
        track_evo_by_race[(year, rnd)] = float(np.median(slopes)) if slopes else -0.02

    # Total laps per race
    total_laps_by_race: dict[tuple[int, int], int] = {}
    for (year, rnd), g in all_laps.groupby(["Year", "Round"]):
        total_laps_by_race[(year, rnd)] = int(g["LapNumber"].max())

    return FeatureContext(
        cum_time_by_lap=cum_time,
        position_at_lap=position_at_lap,
        stint_position_at_lap=stint_position_at_lap,
        pit_loss_by_circuit=pit_loss_by_circuit,
        sc_rate_by_circuit=sc_rate_by_circuit,
        track_evo_by_race=track_evo_by_race,
        total_laps_by_race=total_laps_by_race,
    )


def _gap_to_neighbour(
    ctx: FeatureContext,
    year: int,
    rnd: int,
    driver: str,
    pit_lap: int,
    target_position: int,
    all_laps: pd.DataFrame,
) -> tuple[float, str | None]:
    """Cumulative-race-time gap from `driver` to the car currently in
    `target_position` at lap `pit_lap`. Returns (gap_seconds, neighbour_code).

    Positive gap = `driver` is behind the neighbour (typical for car-ahead).
    """
    same_race = all_laps[
        (all_laps["Year"] == year) & (all_laps["Round"] == rnd) & (all_laps["LapNumber"] == pit_lap)
    ]
    if same_race.empty:
        return (np.nan, None)
    me = same_race[same_race["Driver"] == driver]
    other = same_race[same_race["Position"] == target_position]
    if me.empty or other.empty:
        return (np.nan, None)
    other_row = other.iloc[0]
    neighbour = str(other_row["Driver"])
    my_t = ctx.cum_time_by_lap.get((year, rnd, driver, pit_lap), np.nan)
    other_t = ctx.cum_time_by_lap.get((year, rnd, neighbour, pit_lap), np.nan)
    if np.isnan(my_t) or np.isnan(other_t):
        return (np.nan, neighbour)
    return (float(my_t - other_t), neighbour)


def _current_deg_slope(
    all_laps: pd.DataFrame,
    year: int,
    rnd: int,
    driver: str,
    pit_lap: int,
    window: int = 5,
) -> float:
    """OLS slope of fuel-corrected lap time vs. stint position over the
    `window` clean laps immediately preceding `pit_lap`.

    Returns NaN if fewer than 3 clean laps are available.
    """
    g = all_laps[
        (all_laps["Year"] == year)
        & (all_laps["Round"] == rnd)
        & (all_laps["Driver"] == driver)
        & (all_laps["LapNumber"] < pit_lap)
    ]
    if g.empty:
        return float("nan")
    clean = g[g.get("IsCleanLap", True) & (g["StintPosition"] > 1)]
    if len(clean) == 0:
        return float("nan")
    recent = clean.sort_values("LapNumber").tail(window)
    if len(recent) < 3:
        return float("nan")
    x = recent["StintPosition"].to_numpy(dtype=float)
    y = recent["LapTimeFuelCorrected"].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(x[mask], y[mask], 1)
    return float(slope)


def build_features(
    all_laps: pd.DataFrame,
    pits: pd.DataFrame,
    deg_model: DegradationModel,
) -> pd.DataFrame:
    """Build the real undercut feature matrix.

    Returns a DataFrame with all FEATURE_COLS populated from data plus:
      - label
      - group_id (for grouped CV)
      - year, round_num, circuit_name, driver_code, pit_lap  (for joins)
    """
    ctx = _build_context(all_laps, pits)
    rows: list[dict] = []

    for _, p in pits.iterrows():
        year = int(p["year"])
        rnd = int(p["round_num"])
        drv = str(p["driver_code"])
        pit_lap = int(p["pit_lap"])
        circuit = str(p["circuit_name"]) if "circuit_name" in p else ""

        race_laps = all_laps[
            (all_laps["Year"] == year) & (all_laps["Round"] == rnd) & (all_laps["Driver"] == drv)
        ]
        if race_laps.empty:
            continue
        pit_row = race_laps[race_laps["LapNumber"] == pit_lap]
        if pit_row.empty:
            continue

        pos_before = pit_row["Position"].iloc[0]
        post = race_laps[race_laps["LapNumber"] == pit_lap + 5]
        pos_after = post["Position"].iloc[0] if not post.empty else None
        if pd.isna(pos_before) or pos_after is None or pd.isna(pos_after):
            continue
        pos_before = int(pos_before)
        pos_after = int(pos_after)

        # Compound + tyre age from the laps we already enriched
        compound = (
            pit_row["Compound"].iloc[0] if pd.notna(pit_row["Compound"].iloc[0]) else "MEDIUM"
        )
        tyre_age = (
            int(pit_row["StintPosition"].iloc[0])
            if pd.notna(pit_row["StintPosition"].iloc[0])
            else 1
        )

        # Real gap features
        gap_ahead_s, ahead_drv = _gap_to_neighbour(
            ctx, year, rnd, drv, pit_lap, pos_before - 1, all_laps
        )
        gap_behind_s, behind_drv = _gap_to_neighbour(
            ctx, year, rnd, drv, pit_lap, pos_before + 1, all_laps
        )

        # Tyre age delta vs neighbours (stint position of the other driver on the same lap)
        ahead_age = (
            ctx.stint_position_at_lap.get((year, rnd, ahead_drv, pit_lap), np.nan)
            if ahead_drv
            else np.nan
        )
        behind_age = (
            ctx.stint_position_at_lap.get((year, rnd, behind_drv, pit_lap), np.nan)
            if behind_drv
            else np.nan
        )
        tyre_age_delta_vs_ahead = (tyre_age - ahead_age) if not np.isnan(ahead_age) else 0.0
        tyre_age_delta_vs_behind = (tyre_age - behind_age) if not np.isnan(behind_age) else 0.0

        # Current degradation slope from the last 5 clean laps
        cur_slope = _current_deg_slope(all_laps, year, rnd, drv, pit_lap)
        if np.isnan(cur_slope):
            cur_slope = 0.08  # fallback only when the stint had too few clean laps

        # Expected pace gain from a fresh tyre (delta from the degradation model)
        try:
            pace_delta = float(deg_model.predict_delta(compound, circuit, np.array([tyre_age]))[0])
            if np.isnan(pace_delta):
                pace_delta = cur_slope * tyre_age
        except Exception:
            pace_delta = cur_slope * tyre_age

        total_laps = ctx.total_laps_by_race.get((year, rnd), pit_lap + 20)
        laps_remaining = max(total_laps - pit_lap, 1)
        race_progress = float(pit_lap) / max(total_laps, 1)

        rows.append(
            {
                "year": year,
                "round_num": rnd,
                "circuit_name": circuit,
                "driver_code": drv,
                "pit_lap": pit_lap,
                # Features
                "gap_ahead_s": float(gap_ahead_s) if not np.isnan(gap_ahead_s) else 99.0,
                "gap_behind_s": float(gap_behind_s) if not np.isnan(gap_behind_s) else 99.0,
                "tyre_age": tyre_age,
                "tyre_age_delta_vs_ahead": float(tyre_age_delta_vs_ahead),
                "tyre_age_delta_vs_behind": float(tyre_age_delta_vs_behind),
                "compound_idx": COMPOUND_IDX.get(compound, 1),
                "current_deg_slope": float(cur_slope),
                "expected_fresh_pace_delta": pace_delta,
                "laps_remaining": laps_remaining,
                "race_progress_pct": race_progress,
                "sc_prob_next_5": ctx.sc_rate_by_circuit.get(circuit, 0.05),
                "pit_loss_circuit_s": ctx.pit_loss_by_circuit.get(circuit, 22.0),
                "track_evolution_s_per_lap": ctx.track_evo_by_race.get((year, rnd), -0.02),
                # Label
                "label": int(pos_after < pos_before),
                "group_id": f"{year}_{rnd:02d}",
            }
        )

    return pd.DataFrame(rows)
